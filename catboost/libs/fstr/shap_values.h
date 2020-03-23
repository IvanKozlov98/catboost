#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/types.h>
#include <util/ysaveload.h>


struct TShapValue {
    int Feature = -1;
    TVector<double> Value;

public:
    TShapValue() = default;

    TShapValue(int feature, int approxDimension)
        : Feature(feature)
        , Value(approxDimension)
    {
    }

    Y_SAVELOAD_DEFINE(Feature, Value);
};

struct TFixedFeatureParams {
    enum class EMode {
        FixedOn, FixedOff, NotFixed
    };

    int Feature = -1;
    EMode FixedFeatureMode = EMode::NotFixed;

public:
    TFixedFeatureParams() = default;

    TFixedFeatureParams(int feature, EMode fixedFeatureMode)
        : Feature(feature)
        , FixedFeatureMode(fixedFeatureMode)
        {
        }
};


struct TShapPreparedTrees {
    TVector<TVector<TVector<TShapValue>>> ShapValuesByLeafForAllTrees; // [treeIdx][leafIdx][shapFeature] trees * 2^d * d
    TVector<TVector<TVector<TVector<TVector<double>>>>> ShapInteractionValuesByLeafForAllTrees; // [treeIdx][leafIdx][dim][feature1][feature2]
    TVector<TVector<double>> MeanValuesForAllTrees;
    TVector<double> AverageApproxByTree;
    TVector<int> BinFeatureCombinationClass;
    TVector<TVector<int>> CombinationClassFeatures;
    bool CalcShapValuesByLeafForAllTrees;
    bool CalcInternalValues;
    TVector<double> LeafWeightsForAllTrees;
    TVector<TVector<TVector<double>>> SubtreeWeightsForAllTrees;

public:
    TShapPreparedTrees() = default;

    TShapPreparedTrees(
        const TVector<TVector<TVector<TShapValue>>>& shapValuesByLeafForAllTrees,
        const TVector<TVector<double>>& meanValuesForAllTrees
    )
        : ShapValuesByLeafForAllTrees(shapValuesByLeafForAllTrees)
        , MeanValuesForAllTrees(meanValuesForAllTrees)
    {
    }

    Y_SAVELOAD_DEFINE(
        ShapValuesByLeafForAllTrees,
        MeanValuesForAllTrees,
        AverageApproxByTree,
        BinFeatureCombinationClass,
        CombinationClassFeatures,
        CalcShapValuesByLeafForAllTrees,
        CalcInternalValues,
        LeafWeightsForAllTrees,
        SubtreeWeightsForAllTrees
    );
};

void CalcShapValuesForDocumentMulti(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int flatFeatureCount,
    TConstArrayRef<NCB::NModelEvaluation::TCalcerIndexType> docIndexes,
    size_t documentIdx,
    TVector<TVector<double>>* shapValues
);

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor
);

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    const NCB::TDataProvider* dataset, // can be nullptr if model has LeafWeights
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    bool calcInternalValues = false
);

void CalcShapValuesByLeaf(
    const TFullModel& model,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    bool calcInternalValues,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees 
);

// returned: ShapValues[documentIdx][dimension][feature]
TVector<TVector<TVector<double>>> CalcShapValuesMulti(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor
);

// returned: ShapValues[documentIdx][feature]
TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor
);

// returned: ShapInteractionValues[featureIdx1][featureIdx2][dim][documentIdx]
TVector<TVector<TVector<TVector<double>>>> CalcShapInteractionValuesMulti(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor
);

// outputs for each document in order for each dimension in order an array of feature contributions
void CalcAndOutputShapValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TString& outputPath,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor
);

void CalcShapValuesInternalForFeature(
    const TShapPreparedTrees& preparedTrees,
    const TFullModel& model,
    int logPeriod,
    ui32 start,
    ui32 end,
    ui32 featuresCount,
    const NCB::TObjectsDataProvider& objectsData,
    TVector<TVector<TVector<double>>>* shapValues, // [docIdx][featureIdx][dim]
    NPar::TLocalExecutor* localExecutor
);
