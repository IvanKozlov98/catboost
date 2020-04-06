#include "shap_values.h"

#include "util.h"

#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>
#include <catboost/libs/model/cpu/quantization.h>


using namespace NCB;


namespace {
    struct TFeaturePathElement {
        int Feature;
        double ZeroPathsFraction;
        double OnePathsFraction;
        double Weight;

        TFeaturePathElement() = default;

        TFeaturePathElement(int feature, double zeroPathsFraction, double onePathsFraction, double weight)
            : Feature(feature)
            , ZeroPathsFraction(zeroPathsFraction)
            , OnePathsFraction(onePathsFraction)
            , Weight(weight)
        {
        }
    };
} //anonymous

static TVector<TFeaturePathElement> ExtendFeaturePath(
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature
) {
    const size_t pathLength = oldFeaturePath.size();

    TVector<TFeaturePathElement> newFeaturePath(pathLength + 1);
    Copy(oldFeaturePath.begin(), oldFeaturePath.begin() + pathLength, newFeaturePath.begin());

    const double weight = pathLength == 0 ? 1.0 : 0.0;
    newFeaturePath[pathLength] = TFeaturePathElement(feature, zeroPathsFraction, onePathsFraction, weight);

    for (int elementIdx = pathLength - 1; elementIdx >= 0; --elementIdx) {
        newFeaturePath[elementIdx + 1].Weight += onePathsFraction * newFeaturePath[elementIdx].Weight * (elementIdx + 1) / (pathLength + 1);
        newFeaturePath[elementIdx].Weight = zeroPathsFraction * newFeaturePath[elementIdx].Weight * (pathLength - elementIdx) / (pathLength + 1);
    }

    return newFeaturePath;
}

static TVector<TFeaturePathElement> UnwindFeaturePath(
    const TVector<TFeaturePathElement>& oldFeaturePath,
    size_t eraseElementIdx)
{
    const size_t pathLength = oldFeaturePath.size();
    CB_ENSURE(pathLength > 0, "Path to unwind must have at least one element");

    TVector<TFeaturePathElement> newFeaturePath(
        oldFeaturePath.begin(),
        oldFeaturePath.begin() + pathLength - 1);

    for (size_t elementIdx = eraseElementIdx; elementIdx < pathLength - 1; ++elementIdx) {
        newFeaturePath[elementIdx].Feature = oldFeaturePath[elementIdx + 1].Feature;
        newFeaturePath[elementIdx].ZeroPathsFraction = oldFeaturePath[elementIdx + 1].ZeroPathsFraction;
        newFeaturePath[elementIdx].OnePathsFraction = oldFeaturePath[elementIdx + 1].OnePathsFraction;
    }

    const double onePathsFraction = oldFeaturePath[eraseElementIdx].OnePathsFraction;
    const double zeroPathsFraction = oldFeaturePath[eraseElementIdx].ZeroPathsFraction;
    double weightDiff = oldFeaturePath[pathLength - 1].Weight;

    if (!FuzzyEquals(1 + onePathsFraction, 1 + 0.0)) {
        for (int elementIdx = pathLength - 2; elementIdx >= 0; --elementIdx) {
            double oldWeight = newFeaturePath[elementIdx].Weight;
            newFeaturePath[elementIdx].Weight = weightDiff * pathLength
                / (onePathsFraction * (elementIdx + 1));
            weightDiff = oldWeight
                - newFeaturePath[elementIdx].Weight * zeroPathsFraction * (pathLength - elementIdx - 1)
                    / pathLength;
        }
    } else {
        for (int elementIdx = pathLength - 2; elementIdx >= 0; --elementIdx) {
            newFeaturePath[elementIdx].Weight *= pathLength
                / (zeroPathsFraction * (pathLength - elementIdx - 1));
        }
    }

    return newFeaturePath;
}

static void UpdateShapByFeaturePath(
    const TVector<TFeaturePathElement>& featurePath,
    const double* leafValuesPtr,
    size_t leafId,
    int approxDimension,
    bool isOblivious,
    double averageTreeApprox,
    double conditionFeatureFraction,
    TVector<TShapValue>* shapValuesInternal
) {
    const int approxDimOffset = isOblivious ? approxDimension : 1;
    for (size_t elementIdx = 1; elementIdx < featurePath.size(); ++elementIdx) {
        const TVector<TFeaturePathElement> unwoundPath = UnwindFeaturePath(featurePath, elementIdx);
        double weightSum = 0.0;
        for (const TFeaturePathElement& unwoundPathElement : unwoundPath) {
            weightSum += unwoundPathElement.Weight;
        }
        const TFeaturePathElement& element = featurePath[elementIdx];
        const auto sameFeatureShapValue = FindIf(
            shapValuesInternal->begin(),
            shapValuesInternal->end(),
            [element](const TShapValue& shapValue) {
                return shapValue.Feature == element.Feature;
            }
        );
        const double coefficient =
            conditionFeatureFraction * weightSum * (element.OnePathsFraction - element.ZeroPathsFraction);
        if (sameFeatureShapValue == shapValuesInternal->end()) {
            shapValuesInternal->emplace_back(element.Feature, approxDimension);
            for (int dimension = 0; dimension < approxDimension; ++dimension) {
                double value = coefficient * (leafValuesPtr[leafId * approxDimOffset + dimension] - averageTreeApprox);
                shapValuesInternal->back().Value[dimension] = value;
            }
        } else {
            for (int dimension = 0; dimension < approxDimension; ++dimension) {
                double addValue = coefficient * (leafValuesPtr[leafId * approxDimOffset + dimension] - averageTreeApprox);
                sameFeatureShapValue->Value[dimension] += addValue;
            }
        }
    }
}

TConditionsFeatureFraction::TConditionsFeatureFraction(
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int combinationClass,
    double conditionFeatureFraction,
    double hotCoefficient,
    double coldCoefficient
) {
    HotConditionFeatureFraction = conditionFeatureFraction;
    ColdConditionFeatureFraction = conditionFeatureFraction;
    if (fixedFeatureParams.Defined() && combinationClass == fixedFeatureParams->Feature) {
        switch (fixedFeatureParams->FixedFeatureMode) {
            case TFixedFeatureParams::EMode::FixedOn: {
                ColdConditionFeatureFraction = 0;
                break;
            }
            case TFixedFeatureParams::EMode::FixedOff: {
                HotConditionFeatureFraction *= hotCoefficient;
                ColdConditionFeatureFraction *= coldCoefficient;
                break;
            }
            default: {
                Y_UNREACHABLE();
            }
        }
    }
}

static void ExtendFeaturePathIfFeatureNotFixed(
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature,
    TVector<TFeaturePathElement>* featurePath
) {
    if (!fixedFeatureParams.Defined() || 
        (fixedFeatureParams->FixedFeatureMode == TFixedFeatureParams::EMode::NotFixed || fixedFeatureParams->Feature != feature)) {
        *featurePath = ExtendFeaturePath(
            oldFeaturePath,
            zeroPathsFraction,
            onePathsFraction,
            feature);
    } else {
        const size_t pathLength = oldFeaturePath.size();
        *featurePath = TVector<TFeaturePathElement>(oldFeaturePath.begin(), oldFeaturePath.begin() + pathLength);
    }
}

static void CalcObliviousInternalShapValuesForLeafRecursive(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t documentLeafIdx,
    size_t treeIdx,
    int depth,
    const TVector<TVector<double>>& subtreeWeights,
    size_t nodeIdx,
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature,
    bool calcInternalValues,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const double conditionFeatureFraction,
    TVector<TShapValue>* shapValuesInternal,
    double averageTreeApprox
) {
    if (FuzzyEquals(1.0 + conditionFeatureFraction, 1.0 + 0.0)) {
        return;
    }

    TVector<TFeaturePathElement> featurePath;
    ExtendFeaturePathIfFeatureNotFixed(
        fixedFeatureParams,
        oldFeaturePath,
        zeroPathsFraction,
        onePathsFraction,
        feature,
        &featurePath
    );

    if (depth == forest.GetTreeSizes()[treeIdx]) {
        UpdateShapByFeaturePath(
            featurePath,
            forest.GetFirstLeafPtrForTree(treeIdx),
            nodeIdx,
            forest.GetDimensionsCount(),
            /*isOblivious*/ true,
            averageTreeApprox,
            conditionFeatureFraction,
            shapValuesInternal
        );
    } else {
        double newZeroPathsFraction = 1.0;
        double newOnePathsFraction = 1.0;

        const size_t remainingDepth = forest.GetTreeSizes()[treeIdx] - depth - 1;
        const int combinationClass = binFeatureCombinationClass[
            forest.GetTreeSplits()[forest.GetTreeStartOffsets()[treeIdx] + remainingDepth]
        ];

        const auto sameFeatureElement = FindIf(
            featurePath.begin(),
            featurePath.end(),
            [combinationClass](const TFeaturePathElement& element) {
                return element.Feature == combinationClass;
            }
        );

        if (sameFeatureElement != featurePath.end()) {
            const size_t sameFeatureIndex = sameFeatureElement - featurePath.begin();
            newZeroPathsFraction = featurePath[sameFeatureIndex].ZeroPathsFraction;
            newOnePathsFraction = featurePath[sameFeatureIndex].OnePathsFraction;
            featurePath = UnwindFeaturePath(featurePath, sameFeatureIndex);
        }

        const bool isGoRight = (documentLeafIdx >> remainingDepth) & 1;
        const size_t goNodeIdx = nodeIdx * 2 + isGoRight;
        const size_t skipNodeIdx = nodeIdx * 2 + !isGoRight;
        const double hotCoefficient = subtreeWeights[depth + 1][goNodeIdx] / subtreeWeights[depth][nodeIdx];
        const double coldCoefficient = subtreeWeights[depth + 1][skipNodeIdx] / subtreeWeights[depth][nodeIdx];
        TConditionsFeatureFraction conditionsFeatureFraction{
            fixedFeatureParams,
            combinationClass,
            conditionFeatureFraction,
            hotCoefficient,
            coldCoefficient
        };

        if (!FuzzyEquals(1 + subtreeWeights[depth + 1][goNodeIdx], 1 + 0.0)) {
            double newZeroPathsFractionGoNode = newZeroPathsFraction * hotCoefficient;
            CalcObliviousInternalShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                documentLeafIdx,
                treeIdx,
                depth + 1,
                subtreeWeights,
                goNodeIdx,
                featurePath,
                newZeroPathsFractionGoNode,
                newOnePathsFraction,
                combinationClass,
                calcInternalValues,
                fixedFeatureParams,
                conditionsFeatureFraction.HotConditionFeatureFraction,
                shapValuesInternal,
                averageTreeApprox
            );
        }

        if (!FuzzyEquals(1 + subtreeWeights[depth + 1][skipNodeIdx], 1 + 0.0)) {
            double newZeroPathsFractionSkipNode = newZeroPathsFraction * coldCoefficient;
            CalcObliviousInternalShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                documentLeafIdx,
                treeIdx,
                depth + 1,
                subtreeWeights,
                skipNodeIdx,
                featurePath,
                newZeroPathsFractionSkipNode,
                /*onePathFraction*/ 0,
                combinationClass,
                calcInternalValues,
                fixedFeatureParams,
                conditionsFeatureFraction.ColdConditionFeatureFraction,
                shapValuesInternal,
                averageTreeApprox
            );
        }
    }
}

static void CalcNonObliviousInternalShapValuesForLeafRecursive(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<bool>& mapNodeIdToIsGoRight,
    size_t treeIdx,
    int depth,
    const TVector<TVector<double>>& subtreeWeights,
    size_t nodeIdx,
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const double conditionFeatureFraction,
    bool calcInternalValues,
    TVector<TShapValue>* shapValuesInternal,
    double averageTreeApprox
) {
    if (FuzzyEquals(1.0 + conditionFeatureFraction, 1.0 + 0.0)) {
        return;
    }

    TVector<TFeaturePathElement> featurePath;
    ExtendFeaturePathIfFeatureNotFixed(
        fixedFeatureParams,
        oldFeaturePath,
        zeroPathsFraction,
        onePathsFraction,
        feature,
        &featurePath
    );

    const auto& node = forest.GetNonSymmetricStepNodes()[nodeIdx];
    const size_t startOffset = forest.GetTreeStartOffsets()[treeIdx];
    size_t goNodeIdx;
    size_t skipNodeIdx;
    if (mapNodeIdToIsGoRight[nodeIdx - startOffset]) {
        goNodeIdx = nodeIdx + node.RightSubtreeDiff;
        skipNodeIdx = nodeIdx + node.LeftSubtreeDiff;
    } else {
        goNodeIdx = nodeIdx + node.LeftSubtreeDiff;
        skipNodeIdx = nodeIdx + node.RightSubtreeDiff;
    }
    // goNodeIdx == nodeIdx mean that nodeIdx is a terminal node for
    // observed object. That's why we should update shap here.
    // Similary for skipNodeIdx.
    if (goNodeIdx == nodeIdx || skipNodeIdx == nodeIdx) {
        UpdateShapByFeaturePath(
            featurePath,
            &forest.GetLeafValues()[0],
            forest.GetNonSymmetricNodeIdToLeafId()[nodeIdx],
            forest.GetDimensionsCount(),
            /*isOblivious*/ false,
            averageTreeApprox,
            conditionFeatureFraction,
            shapValuesInternal
        );
    }
    double newZeroPathsFraction = 1.0;
    double newOnePathsFraction = 1.0;

    const int combinationClass = binFeatureCombinationClass[
        forest.GetTreeSplits()[nodeIdx]
    ];

    const auto sameFeatureElement = FindIf(
        featurePath.begin(),
        featurePath.end(),
        [combinationClass](const TFeaturePathElement& element) {
            return element.Feature == combinationClass;
        }
    );

    if (sameFeatureElement != featurePath.end()) {
        const size_t sameFeatureIndex = sameFeatureElement - featurePath.begin();
        newZeroPathsFraction = featurePath[sameFeatureIndex].ZeroPathsFraction;
        newOnePathsFraction = featurePath[sameFeatureIndex].OnePathsFraction;
        featurePath = UnwindFeaturePath(featurePath, sameFeatureIndex);
    }
    
    const double hotCoefficient = goNodeIdx != nodeIdx ? 
        subtreeWeights[0][goNodeIdx - startOffset] / subtreeWeights[0][nodeIdx - startOffset] : -1.0;
    const double coldCoefficient = skipNodeIdx != nodeIdx ? 
        subtreeWeights[0][skipNodeIdx - startOffset] / subtreeWeights[0][nodeIdx - startOffset] : -1.0;
    TConditionsFeatureFraction conditionsFeatureFraction{
        fixedFeatureParams,
        combinationClass,
        conditionFeatureFraction,
        hotCoefficient,
        coldCoefficient
    };
    
    if (goNodeIdx != nodeIdx && !FuzzyEquals(1 + subtreeWeights[0][goNodeIdx - startOffset], 1 + 0.0)) {
        double newZeroPathsFractionGoNode = newZeroPathsFraction * hotCoefficient;
        CalcNonObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            depth + 1,
            subtreeWeights,
            goNodeIdx,
            featurePath,
            newZeroPathsFractionGoNode,
            newOnePathsFraction,
            combinationClass,
            fixedFeatureParams,
            conditionsFeatureFraction.HotConditionFeatureFraction,
            calcInternalValues,
            shapValuesInternal,
            averageTreeApprox
        );
    }

    if (skipNodeIdx != nodeIdx && !FuzzyEquals(1 + subtreeWeights[0][skipNodeIdx - startOffset], 1 + 0.0)) {
        double newZeroPathsFractionSkipNode = newZeroPathsFraction * coldCoefficient;
        CalcNonObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            depth + 1,
            subtreeWeights,
            skipNodeIdx,
            featurePath,
            newZeroPathsFractionSkipNode,
            /*onePathFraction*/ 0,
            combinationClass,
            fixedFeatureParams,
            conditionsFeatureFraction.ColdConditionFeatureFraction,
            calcInternalValues,
            shapValuesInternal,
            averageTreeApprox
        );
    }
}

static void UnpackInternalShaps(const TVector<TShapValue>& shapValuesInternal, const TVector<TVector<int>>& combinationClassFeatures,  TVector<TShapValue>* shapValues) {
    shapValues->clear();
    if (shapValuesInternal.empty()) {
        return;
    }
    const int approxDimension = shapValuesInternal[0].Value.ysize();
    for (const auto & shapValueInternal: shapValuesInternal) {
        const TVector<int> &flatFeatures = combinationClassFeatures[shapValueInternal.Feature];

        for (int flatFeatureIdx : flatFeatures) {
            const auto sameFeatureShapValue = FindIf(
                shapValues->begin(),
                shapValues->end(),
                [flatFeatureIdx](const TShapValue &shapValue) {
                    return shapValue.Feature == flatFeatureIdx;
                }
            );
            double coefficient = flatFeatures.size();
            if (sameFeatureShapValue == shapValues->end()) {
                shapValues->emplace_back(flatFeatureIdx, approxDimension);
                for (int dimension = 0; dimension < approxDimension; ++dimension) {
                    double value = shapValueInternal.Value[dimension] / coefficient;
                    shapValues->back().Value[dimension] = value;
                }
            } else {
                for (int dimension = 0; dimension < approxDimension; ++dimension) {
                    double addValue = shapValueInternal.Value[dimension] / coefficient;
                    sameFeatureShapValue->Value[dimension] += addValue;
                }
            }
        }
    }
}

static inline void CalcObliviousShapValuesForLeaf(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t documentLeafIdx,
    size_t treeIdx,
    const TVector<TVector<double>>& subtreeWeights,
    bool calcInternalValues,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    TVector<TShapValue>* shapValues,
    double averageTreeApprox
) {
    shapValues->clear();

    if (calcInternalValues) {
        CalcObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            documentLeafIdx,
            treeIdx,
            /*depth*/ 0,
            subtreeWeights,
            /*nodeIdx*/ 0,
            /*initialFeaturePath*/ {},
            /*zeroPathFraction*/ 1,
            /*onePathFraction*/ 1,
            /*feature*/ -1,
            calcInternalValues,
            fixedFeatureParams,
            /*conditionFeatureFraction*/ 1,
            shapValues,
            averageTreeApprox
        );
    } else {
        TVector<TShapValue> shapValuesInternal;
        CalcObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            documentLeafIdx,
            treeIdx,
            /*depth*/ 0,
            subtreeWeights,
            /*nodeIdx*/ 0,
            /*initialFeaturePath*/ {},
            /*zeroPathFraction*/ 1,
            /*onePathFraction*/ 1,
            /*feature*/ -1,
            calcInternalValues,
            fixedFeatureParams,
            /*conditionFeatureFraction*/ 1,
            &shapValuesInternal,
            averageTreeApprox
        );
        UnpackInternalShaps(shapValuesInternal, combinationClassFeatures, shapValues);
    }
}

static inline void CalcNonObliviousShapValuesForLeaf(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<bool>& mapNodeIdToIsGoRight,
    size_t treeIdx,
    const TVector<TVector<double>>& subtreeWeights,
    bool calcInternalValues,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    TVector<TShapValue>* shapValues,
    double averageTreeApprox
) {
    shapValues->clear();

    if (calcInternalValues) {
        CalcNonObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            /*depth*/ 0,
            subtreeWeights,
            /*nodeIdx*/ forest.GetTreeStartOffsets()[treeIdx],
            /*initialFeaturePath*/ {},
            /*zeroPathFraction*/ 1,
            /*onePathFraction*/ 1,
            /*feature*/ -1,
            fixedFeatureParams,
            /*conditionFeatureFraction*/ 1.0,
            calcInternalValues,
            shapValues,
            averageTreeApprox
        );
    } else {
        TVector<TShapValue> shapValuesInternal;
        CalcNonObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            /*depth*/ 0,
            subtreeWeights,
            /*nodeIdx*/ forest.GetTreeStartOffsets()[treeIdx],
            /*initialFeaturePath*/ {},
            /*zeroPathFraction*/ 1,
            /*onePathFraction*/ 1,
            /*feature*/ -1,
            fixedFeatureParams,
            /*conditionFeatureFraction*/ 1.0,
            calcInternalValues,
            &shapValuesInternal,
            averageTreeApprox
        );
        UnpackInternalShaps(shapValuesInternal, combinationClassFeatures, shapValues);
    }
}

// All calculations only for one docId
static TVector<bool> GetDocumentPathToLeafForNonObliviousBlock(
    const TModelTrees& forest,
    const size_t docIdx,
    const size_t treeIdx,
    const NCB::NModelEvaluation::TCPUEvaluatorQuantizedData& block
) {
    const ui8* binFeatures = block.QuantizedData.data();
    const size_t docCountInBlock = block.ObjectsCount;
    const TRepackedBin* treeSplitsPtr = forest.GetRepackedBins().data();
    const auto firstLeafOffsets = forest.GetFirstLeafOffsets();
    const int totalNodesCount = forest.GetTreeSplits().size();
    const bool isLastTree = static_cast<size_t>(treeIdx + 1) == forest.GetTreeStartOffsets().size();
    const size_t endOffset = isLastTree ? totalNodesCount : forest.GetTreeStartOffsets()[treeIdx + 1];
    TVector<bool> mapNodeIdToIsGoRight;
    for (NCB::NModelEvaluation::TCalcerIndexType nodeIdx = forest.GetTreeStartOffsets()[treeIdx]; nodeIdx < endOffset; ++nodeIdx) {
        const TRepackedBin split = treeSplitsPtr[nodeIdx];
        ui8 featureValue = binFeatures[split.FeatureIndex * docCountInBlock + docIdx];
        if (!forest.GetOneHotFeatures().empty()) {
            featureValue ^= split.XorMask;
        }
        mapNodeIdToIsGoRight.push_back(featureValue >= split.SplitIdx);
    }
    return mapNodeIdToIsGoRight;
}

static TVector<bool> GetDocumentIsGoRightMapperForNodesInNonObliviousTree(
    const TModelTrees& forest,
    size_t treeIdx,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    size_t documentIdx
) {
    const NModelEvaluation::TCPUEvaluatorQuantizedData* dataPtr = reinterpret_cast<const NModelEvaluation::TCPUEvaluatorQuantizedData*>(binarizedFeaturesForBlock);
    Y_ASSERT(dataPtr);
    auto blockId = documentIdx / NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE;
    auto subBlock = dataPtr->ExtractBlock(blockId);
    return GetDocumentPathToLeafForNonObliviousBlock(
        forest,
        documentIdx % NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE,
        treeIdx,
        subBlock
    );
}

static void TreeShap(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int featuresCount,
    TConstArrayRef<NModelEvaluation::TCalcerIndexType> docIndices,
    size_t documentIdxInBlock,
    TVector<TVector<double>>* shapValues
) {
    const int approxDimension = model.GetDimensionsCount();
    shapValues->assign(approxDimension, TVector<double>(featuresCount + 1, 0.0));
    const size_t treeCount = model.GetTreeCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        if (preparedTrees.CalcShapValuesByLeafForAllTrees && model.IsOblivious()) {
            Y_ASSERT(docIndices[treeIdx] < preparedTrees.ShapValuesByLeafForAllTrees[treeIdx].size());
            for (const TShapValue& shapValue : preparedTrees.ShapValuesByLeafForAllTrees[treeIdx][docIndices[treeIdx]]) {
                for (int dimension = 0; dimension < approxDimension; ++dimension) {
                    (*shapValues)[dimension][shapValue.Feature] += shapValue.Value[dimension];
                }
            }
        } else {
            TVector<TShapValue> shapValuesByLeaf;
            if (model.IsOblivious()) {
                CalcObliviousShapValuesForLeaf(
                    *model.ModelTrees.Get(),
                    preparedTrees.BinFeatureCombinationClass,
                    preparedTrees.CombinationClassFeatures,
                    docIndices[treeIdx],
                    treeIdx,
                    preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                    preparedTrees.CalcInternalValues,
                    fixedFeatureParams,
                    &shapValuesByLeaf,
                    preparedTrees.AverageApproxByTree[treeIdx]
                );
            } else {
                TVector<bool> mapNodeIdToIsGoRight = GetDocumentIsGoRightMapperForNodesInNonObliviousTree(
                    *model.ModelTrees.Get(),
                    treeIdx,
                    binarizedFeaturesForBlock,
                    documentIdxInBlock
                );
                CalcNonObliviousShapValuesForLeaf(
                    *model.ModelTrees.Get(),
                    preparedTrees.BinFeatureCombinationClass,
                    preparedTrees.CombinationClassFeatures,
                    mapNodeIdToIsGoRight,
                    treeIdx,
                    preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                    preparedTrees.CalcInternalValues,
                    fixedFeatureParams,
                    &shapValuesByLeaf,
                    preparedTrees.AverageApproxByTree[treeIdx]
                );
            }
            for (const TShapValue& shapValue : shapValuesByLeaf) {
                for (int dimension = 0; dimension < approxDimension; ++dimension) {
                    (*shapValues)[dimension][shapValue.Feature] += shapValue.Value[dimension];
                }
            }
        }

        for (int dimension = 0; dimension < approxDimension; ++dimension) {
            (*shapValues)[dimension][featuresCount] +=
                    preparedTrees.MeanValuesForAllTrees[treeIdx][dimension];
        }
    }
    if (approxDimension == 1) {
        (*shapValues)[0][featuresCount] += model.GetScaleAndBias().Bias;
    }
}

void CalcShapValuesForDocumentMulti(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int featuresCount,
    TConstArrayRef<NModelEvaluation::TCalcerIndexType> docIndices,
    ECalcTypeShapValues calcType,
    size_t documentIdxInBlock,
    size_t documentIdx,
    TVector<TVector<double>>* shapValues
) {
    switch (calcType) {
        case ECalcTypeShapValues::IndependentTreeSHAP: {
            IndependentTreeShap(
                model,
                preparedTrees,
                featuresCount,
                docIndices,
                documentIdx,
                shapValues
            );
            break;
        }
        case ECalcTypeShapValues::TreeSHAP: {
            TreeShap(
                model,
                preparedTrees,
                binarizedFeaturesForBlock,
                fixedFeatureParams,
                featuresCount,
                docIndices,
                documentIdxInBlock,
                shapValues
            );
            break;
        }
        default: {
            CB_ENSURE_INTERNAL(false, "This mode of calculation shap values not implemented");
        }
    }
}

void CalcShapValuesForDocumentMulti(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    int featuresCount,
    TConstArrayRef<NModelEvaluation::TCalcerIndexType> docIndices,
    size_t documentIdxInBlock,
    TVector<TVector<double>>* shapValues
) {
    CalcShapValuesForDocumentMulti(
        model,
        preparedTrees,
        binarizedFeaturesForBlock,
        /*fixedFeatureParams*/ Nothing(),
        featuresCount,
        docIndices,
        ECalcTypeShapValues::TreeSHAP,
        documentIdxInBlock,
        /*documentIdx*/ -1, // algorithm TreeSHAP doesn't use this parameter
        shapValues
    );
}

static void CalcShapValuesForDocumentBlockMulti(
    const TFullModel& model,
    const IFeaturesBlockIterator& featuresBlockIterator,
    int flatFeatureCount,
    const TShapPreparedTrees& preparedTrees,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    ECalcTypeShapValues calcType,
    size_t start,
    size_t end,
    NPar::TLocalExecutor* localExecutor,
    TVector<TVector<TVector<double>>>* shapValuesForAllDocuments
) {
    CheckNonZeroApproxForZeroWeightLeaf(model);

    const size_t documentCount = end - start;

    auto binarizedFeaturesForBlock = MakeQuantizedFeaturesForEvaluator(model, featuresBlockIterator, start, end);

    TVector<NModelEvaluation::TCalcerIndexType> indices(binarizedFeaturesForBlock->GetObjectsCount() * model.GetTreeCount());
    model.GetCurrentEvaluator()->CalcLeafIndexes(binarizedFeaturesForBlock.Get(), 0, model.GetTreeCount(), indices);

    const int oldShapValuesSize = shapValuesForAllDocuments->size();
    shapValuesForAllDocuments->resize(oldShapValuesSize + end - start);

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, documentCount);
    localExecutor->ExecRange([&] (size_t documentIdxInBlock) {
        TVector<TVector<double>>& shapValues = (*shapValuesForAllDocuments)[oldShapValuesSize + documentIdxInBlock];

        CalcShapValuesForDocumentMulti(
            model,
            preparedTrees,
            binarizedFeaturesForBlock.Get(),
            fixedFeatureParams,
            flatFeatureCount,
            MakeArrayRef(indices.data() + documentIdxInBlock * model.GetTreeCount(), model.GetTreeCount()),
            calcType,
            documentIdxInBlock,
            /*documentIdx*/ documentIdxInBlock + start,
            &shapValues
        );

    }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void CalcTreeShapValuesByLeafForTreeBlock(
    const TModelTrees& forest,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    size_t treeIdx,
    bool calcInternalValues,
    TShapPreparedTrees* preparedTrees
) {
    const size_t leafCount = (size_t(1) << forest.GetTreeSizes()[treeIdx]);
    TVector<TVector<TShapValue>>& shapValuesByLeaf = preparedTrees->ShapValuesByLeafForAllTrees[treeIdx];
    shapValuesByLeaf.resize(leafCount);
    for (size_t leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        CalcObliviousShapValuesForLeaf(
            forest,
            preparedTrees->BinFeatureCombinationClass,
            preparedTrees->CombinationClassFeatures,
            leafIdx,
            treeIdx,
            preparedTrees->SubtreeWeightsForAllTrees[treeIdx],
            calcInternalValues,
            fixedFeatureParams,
            &shapValuesByLeaf[leafIdx],
            preparedTrees->AverageApproxByTree[treeIdx]
        );
    }
}

static void CalcShapValuesByLeafForTreeBlock(
    const TModelTrees& forest,
    int start,
    int end,
    bool calcInternalValues,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees
) {
    const bool isOblivious = forest.GetNonSymmetricStepNodes().empty() && forest.GetNonSymmetricNodeIdToLeafId().empty();
    NPar::TLocalExecutor::TExecRangeParams blockParams(start, end);
    localExecutor->ExecRange([&] (size_t treeIdx) {
        if (preparedTrees->CalcShapValuesByLeafForAllTrees && isOblivious) {
            switch (calcType) {
                case ECalcTypeShapValues::IndependentTreeSHAP: {
                    CB_ENSURE_INTERNAL(preparedTrees->IndependentTreeShapParams, "Independent params must be defined");
                    CalcIndependentTreeShapValuesByLeafForTreeBlock(
                        forest,
                        treeIdx,
                        preparedTrees
                    );
                    break;
                }
                case ECalcTypeShapValues::TreeSHAP: {
                    CalcTreeShapValuesByLeafForTreeBlock(
                        forest,
                        fixedFeatureParams,
                        treeIdx,
                        calcInternalValues,
                        preparedTrees
                    );
                    break;
                }
                default: {
                    CB_ENSURE_INTERNAL(false, "This mode of calculation shap values not implemented");
                }
            }
        }
    }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
}

void CalcShapValuesByLeaf(
    const TFullModel& model,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    ECalcTypeShapValues calcType,
    int logPeriod,
    bool calcInternalValues,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees 
) {
    const size_t treeCount = model.GetTreeCount();
    const size_t treeBlockSize = CB_THREAD_LIMIT; // least necessary for threading
    TProfileInfo processTreesProfile(treeCount);
    TImportanceLogger treesLogger(treeCount, "trees processed", "Processing trees...", logPeriod);

    for (size_t start = 0; start < treeCount; start += treeBlockSize) {
        size_t end = Min(start + treeBlockSize, treeCount);

        processTreesProfile.StartIterationBlock();

        CalcShapValuesByLeafForTreeBlock(
            *model.ModelTrees,
            start,
            end,
            calcInternalValues,
            fixedFeatureParams,
            calcType,
            localExecutor,
            preparedTrees
        );

        processTreesProfile.FinishIterationBlock(end - start);
        auto profileResults = processTreesProfile.GetProfileResults();
        treesLogger.Log(profileResults);
    }
}

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor
) {
    CB_ENSURE(
        !model.ModelTrees->GetLeafWeights().empty(),
        "Model must have leaf weights or sample pool must be provided"
    );
    TShapPreparedTrees preparedTrees = PrepareTrees(
        model,
        /*dataset*/ nullptr,
        /*referenceDataset*/ nullptr,
        /*mode*/ EPreCalcShapValues::Auto,
        /*calcType*/ ECalcTypeShapValues::TreeSHAP,
        /*modelOutputType*/ EModelOutputType::Raw,
        localExecutor,
        /*calcInternalValues*/ false
    );
    CalcShapValuesByLeaf(
        model,
        /*fixedFeatureParams*/ Nothing(),
        /*calcType*/ ECalcTypeShapValues::TreeSHAP,
        /*logPeriod*/ 0,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees
    );
    return preparedTrees;
}

void CalcShapValuesInternalForFeature(
    const TShapPreparedTrees& preparedTrees,
    const TFullModel& model,
    int /*logPeriod*/,
    ui32 start,
    ui32 end,
    ui32 featuresCount,
    const NCB::TObjectsDataProvider& objectsData,
    TVector<TVector<TVector<double>>>* shapValues, // [docIdx][featureIdx][dim]
    NPar::TLocalExecutor* localExecutor
) {


    CB_ENSURE(start <= end && end <= objectsData.GetObjectCount());
    const TModelTrees& forest = *model.ModelTrees;
    shapValues->clear();
    const ui32 documentCount = end - start;
    shapValues->resize(documentCount);

    THolder<IFeaturesBlockIterator> featuresBlockIterator
        = CreateFeaturesBlockIterator(model, objectsData, start, end);

    const ui32 documentBlockSize = NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE;
    TVector<NModelEvaluation::TCalcerIndexType> indices(documentBlockSize * forest.GetTreeCount());

    for (ui32 startIdx = 0; startIdx < documentCount; startIdx += documentBlockSize) {
        NPar::TLocalExecutor::TExecRangeParams blockParams(startIdx, startIdx + Min(documentBlockSize, documentCount - startIdx));
        featuresBlockIterator->NextBlock(blockParams.LastId - blockParams.FirstId);
        auto binarizedFeaturesForBlock = MakeQuantizedFeaturesForEvaluator(model, *featuresBlockIterator, blockParams.FirstId, blockParams.LastId);


        model.GetCurrentEvaluator()->CalcLeafIndexes(
            binarizedFeaturesForBlock.Get(),
            0, forest.GetTreeCount(),
            MakeArrayRef(indices.data(), binarizedFeaturesForBlock->GetObjectsCount() * forest.GetTreeCount())
        );

        localExecutor->ExecRange([&](ui32 documentIdx) {
            TVector<TVector<double>> &docShapValues = (*shapValues)[documentIdx];
            docShapValues.assign(featuresCount, TVector<double>(forest.GetDimensionsCount() + 1, 0.0));
            auto docIndices = MakeArrayRef(indices.data() + forest.GetTreeCount() * (documentIdx - startIdx), forest.GetTreeCount());
            for (size_t treeIdx = 0; treeIdx < forest.GetTreeCount(); ++treeIdx) {
                if (preparedTrees.CalcShapValuesByLeafForAllTrees && model.IsOblivious()) {
                    for (const TShapValue& shapValue : preparedTrees.ShapValuesByLeafForAllTrees[treeIdx][docIndices[treeIdx]]) {
                        for (int dimension = 0; dimension < (int)forest.GetDimensionsCount(); ++dimension) {
                            docShapValues[shapValue.Feature][dimension] += shapValue.Value[dimension];
                        }
                    }
                } else {
                    TVector<TShapValue> shapValuesByLeaf;

                    if (model.IsOblivious()) {
                        CalcObliviousShapValuesForLeaf(
                            forest,
                            preparedTrees.BinFeatureCombinationClass,
                            preparedTrees.CombinationClassFeatures,
                            docIndices[treeIdx],
                            treeIdx,
                            preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                            preparedTrees.CalcInternalValues,
                            /*fixedFeatureParams*/ Nothing(),
                            &shapValuesByLeaf,
                            preparedTrees.AverageApproxByTree[treeIdx]
                        );
                    } else {
                        const TVector<bool> docPathIndices = GetDocumentIsGoRightMapperForNodesInNonObliviousTree(
                            *model.ModelTrees.Get(),
                            treeIdx,
                            binarizedFeaturesForBlock.Get(),
                            documentIdx - startIdx
                        );
                        CalcNonObliviousShapValuesForLeaf(
                            forest,
                            preparedTrees.BinFeatureCombinationClass,
                            preparedTrees.CombinationClassFeatures,
                            docPathIndices,
                            treeIdx,
                            preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                            preparedTrees.CalcInternalValues,
                            /*fixedFeatureParams*/ Nothing(),
                            &shapValuesByLeaf,
                            preparedTrees.AverageApproxByTree[treeIdx]
                        );
                    }

                    for (const TShapValue& shapValue : shapValuesByLeaf) {
                        for (int dimension = 0; dimension < (int)forest.GetDimensionsCount(); ++dimension) {
                            docShapValues[shapValue.Feature][dimension] += shapValue.Value[dimension];
                        }
                    }
                }
            }
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
    }
}

static TVector<TVector<TVector<double>>> CalcShapValuesWithPreparedTrees(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    ECalcTypeShapValues calcType,
    int logPeriod,
    TShapPreparedTrees* preparedTrees,
    NPar::TLocalExecutor* localExecutor
) {
    const size_t documentCount = dataset.ObjectsGrouping->GetObjectCount();
    const size_t documentBlockSize = CB_THREAD_LIMIT; // least necessary for threading

    const int flatFeatureCount = SafeIntegerCast<int>(dataset.MetaInfo.GetFeatureCount());

    TImportanceLogger documentsLogger(documentCount, "documents processed", "Processing documents...", logPeriod);

    TVector<TVector<TVector<double>>> shapValues;
    shapValues.reserve(documentCount);

    TProfileInfo processDocumentsProfile(documentCount);

    THolder<IFeaturesBlockIterator> featuresBlockIterator
        = CreateFeaturesBlockIterator(model, *dataset.ObjectsData, 0, documentCount);

    for (size_t start = 0; start < documentCount; start += documentBlockSize) {
        size_t end = Min(start + documentBlockSize, documentCount);

        processDocumentsProfile.StartIterationBlock();

        featuresBlockIterator->NextBlock(end - start);

        CalcShapValuesForDocumentBlockMulti(
            model,
            *featuresBlockIterator,
            flatFeatureCount,
            *preparedTrees,
            fixedFeatureParams,
            calcType,
            start,
            end,
            localExecutor,
            &shapValues
        );

        processDocumentsProfile.FinishIterationBlock(end - start);
        auto profileResults = processDocumentsProfile.GetProfileResults();
        documentsLogger.Log(profileResults);
    }

    return shapValues;
}

TVector<TVector<TVector<double>>> CalcShapValuesMulti(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TDataProviderPtr referenceDataset,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    EPreCalcShapValues mode,
    ECalcTypeShapValues calcType,
    EModelOutputType modelOutputType,
    NPar::TLocalExecutor* localExecutor
) {
    TShapPreparedTrees preparedTrees = PrepareTrees(
        model,
        &dataset,
        referenceDataset,
        mode,
        calcType,
        modelOutputType,
        localExecutor,
        /*calcInternalValues*/ false
    );
    CalcShapValuesByLeaf(
        model,
        fixedFeatureParams,
        calcType,
        logPeriod,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees
    );

    return CalcShapValuesWithPreparedTrees(
        model,
        dataset,
        fixedFeatureParams,
        calcType,
        logPeriod,
        &preparedTrees,
        localExecutor
    );
}

TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TDataProviderPtr referenceDataset,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    EPreCalcShapValues mode,
    ECalcTypeShapValues calcType,
    EModelOutputType modelOutputType,
    NPar::TLocalExecutor* localExecutor
) {
    CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1, "Model must not be trained for multiclassification.");
    TVector<TVector<TVector<double>>> shapValuesMulti = CalcShapValuesMulti(
        model,
        dataset,
        referenceDataset,
        fixedFeatureParams,
        logPeriod,
        mode,
        calcType,
        modelOutputType,
        localExecutor
    );

    size_t documentsCount = dataset.ObjectsGrouping->GetObjectCount();
    TVector<TVector<double>> shapValues(documentsCount);

    for (size_t documentIdx = 0; documentIdx < documentsCount; ++documentIdx) {
        shapValues[documentIdx] = std::move(shapValuesMulti[documentIdx][0]);
    }
    return shapValues;
}

static TVector<TVector<TVector<double>>> SwapFeatureAndDocumentAxes(const TVector<TVector<TVector<double>>>& shapValues) {
    const size_t approxDimension = shapValues[0].size();
    const size_t documentCount = shapValues.size();
    const size_t featuresCount = shapValues[0][0].size();
    TVector<TVector<TVector<double>>> swapedShapValues(featuresCount);
    for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
        swapedShapValues[featureIdx].resize(approxDimension);
        for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
            swapedShapValues[featureIdx][dimension].resize(documentCount);
            for (size_t documentIdx = 0; documentIdx < documentCount; ++documentIdx) {
                swapedShapValues[featureIdx][dimension][documentIdx] = shapValues[documentIdx][dimension][featureIdx];
            }
        }
    }
    return swapedShapValues;
}

// returned: ShapValues[featureIdx][dim][documentIdx]
TVector<TVector<TVector<double>>> CalcShapValueWithQuantizedData(
    const TFullModel& model, 
    const TVector<TIntrusivePtr<NModelEvaluation::IQuantizedData>>& quantizedFeatures,
    const TVector<TVector<NModelEvaluation::TCalcerIndexType>>& indices,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const size_t documentCount,
    int logPeriod,
    TShapPreparedTrees* preparedTrees,
    NPar::TLocalExecutor* localExecutor
) {
    CalcShapValuesByLeaf(
        model,
        fixedFeatureParams,
        /*calcType*/ ECalcTypeShapValues::TreeSHAP,
        logPeriod,
        preparedTrees->CalcInternalValues,
        localExecutor,
        preparedTrees
    );
    const TModelTrees& forest = *model.ModelTrees;
    TVector<TVector<TVector<double>>> shapValues(documentCount);
    const int featuresCount = preparedTrees->CombinationClassFeatures.size();
    const size_t documentBlockSize = CB_THREAD_LIMIT;
    for (ui32 startIdx = 0, blockIdx = 0; startIdx < documentCount; startIdx += documentBlockSize, ++blockIdx) {
        NPar::TLocalExecutor::TExecRangeParams blockParams(startIdx, startIdx + Min(documentBlockSize, documentCount - startIdx));
        auto quantizedFeaturesBlock = quantizedFeatures[blockIdx];
        auto& indicesForBlock = indices[blockIdx];
        localExecutor->ExecRange([&](ui32 documentIdx) {
            const size_t documentIdxInBlock = documentIdx - startIdx;
            auto docIndices = MakeArrayRef(indicesForBlock.data() + forest.GetTreeCount() * documentIdxInBlock, forest.GetTreeCount());
            CalcShapValuesForDocumentMulti(
                model,
                *preparedTrees,
                quantizedFeaturesBlock.Get(),
                fixedFeatureParams,
                featuresCount,
                docIndices,
                /*calcType*/ ECalcTypeShapValues::TreeSHAP,
                documentIdxInBlock,
                documentIdx,
                &shapValues[documentIdx]
            );
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
    }

    const auto& swapedShapValues = SwapFeatureAndDocumentAxes(shapValues);
    return swapedShapValues;
}

static void OutputShapValuesMulti(const TVector<TVector<TVector<double>>>& shapValues, TFileOutput& out) {
    for (const auto& shapValuesForDocument : shapValues) {
        for (const auto& shapValuesForClass : shapValuesForDocument) {
            int valuesCount = shapValuesForClass.size();
            for (int valueIdx = 0; valueIdx < valuesCount; ++valueIdx) {
                out << shapValuesForClass[valueIdx] << (valueIdx + 1 == valuesCount ? '\n' : '\t');
            }
        }
    }
}

void CalcAndOutputShapValues(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TString& outputPath,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor
) {
    TShapPreparedTrees preparedTrees = PrepareTrees(
        model,
        &dataset,
        /*referenceDataset*/ nullptr,
        mode,
        /*calcType*/ ECalcTypeShapValues::TreeSHAP,
        /*modelOutputType*/ EModelOutputType::Raw,
        localExecutor,
        /*calcInternalValues*/ false
    );
    CalcShapValuesByLeaf(
        model,
        /*fixedFeatureParams*/ Nothing(),
        /*calcType*/ ECalcTypeShapValues::TreeSHAP,
        logPeriod,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees
    );

    CB_ENSURE_SCALE_IDENTITY(model.GetScaleAndBias(), "SHAP values");
    const int flatFeatureCount = SafeIntegerCast<int>(dataset.MetaInfo.GetFeatureCount());

    const size_t documentCount = dataset.ObjectsGrouping->GetObjectCount();
    const size_t documentBlockSize = CB_THREAD_LIMIT; // least necessary for threading

    TImportanceLogger documentsLogger(documentCount, "documents processed", "Processing documents...", logPeriod);

    TProfileInfo processDocumentsProfile(documentCount);

    THolder<IFeaturesBlockIterator> featuresBlockIterator
        = CreateFeaturesBlockIterator(model, *dataset.ObjectsData, 0, documentCount);

    TFileOutput out(outputPath);
    for (size_t start = 0; start < documentCount; start += documentBlockSize) {
        size_t end = Min(start + documentBlockSize, documentCount);
        processDocumentsProfile.StartIterationBlock();

        TVector<TVector<TVector<double>>> shapValuesForBlock;
        shapValuesForBlock.reserve(end - start);

        featuresBlockIterator->NextBlock(end - start);

        CalcShapValuesForDocumentBlockMulti(
            model,
            *featuresBlockIterator,
            flatFeatureCount,
            preparedTrees,
            /*fixedFeatureParams*/ Nothing(),
            /*calcType*/ ECalcTypeShapValues::TreeSHAP,
            start,
            end,
            localExecutor,
            &shapValuesForBlock
        );

        OutputShapValuesMulti(shapValuesForBlock, out);

        processDocumentsProfile.FinishIterationBlock(end - start);
        auto profileResults = processDocumentsProfile.GetProfileResults();
        documentsLogger.Log(profileResults);
    }
}
