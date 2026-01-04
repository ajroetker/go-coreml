// Package coreml contains protobuf definitions for CoreML model format.
//
// This package provides generated Go types from Apple's CoreML protobuf
// specifications, enabling programmatic construction and serialization
// of CoreML models.
//
// Use go generate to regenerate the protobuf Go code:
//
//	go generate ./...
package coreml

//go:generate protoc --go_out=milspec --go_opt=paths=source_relative MIL.proto
//go:generate protoc --go_out=spec --go_opt=paths=source_relative FeatureTypes.proto DataStructures.proto Parameters.proto Model.proto ArrayFeatureExtractor.proto AudioFeaturePrint.proto BayesianProbitRegressor.proto CategoricalMapping.proto ClassConfidenceThresholding.proto CustomModel.proto DictVectorizer.proto FeatureVectorizer.proto Gazetteer.proto GLMClassifier.proto GLMRegressor.proto Identity.proto Imputer.proto ItemSimilarityRecommender.proto LinkedModel.proto NearestNeighbors.proto NeuralNetwork.proto NonMaximumSuppression.proto Normalizer.proto OneHotEncoder.proto Scaler.proto SoundAnalysisPreprocessing.proto SVM.proto TextClassifier.proto TreeEnsemble.proto VisionFeaturePrint.proto WordEmbedding.proto WordTagger.proto
