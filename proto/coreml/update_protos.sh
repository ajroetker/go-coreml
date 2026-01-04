#!/bin/bash
# update_protos.sh - Download CoreML proto files and add Go package options
#
# Usage: ./update_protos.sh
#
# This script downloads the CoreML protobuf definitions from Apple's coremltools
# repository and adds the necessary go_package options for Go code generation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COREML_BASE_URL="https://raw.githubusercontent.com/apple/coremltools/main/mlmodel/format"

# List of proto files to download
PROTO_FILES=(
    "MIL.proto"
    "Model.proto"
    "FeatureTypes.proto"
    "DataStructures.proto"
    "Parameters.proto"
    "ArrayFeatureExtractor.proto"
    "AudioFeaturePrint.proto"
    "BayesianProbitRegressor.proto"
    "CategoricalMapping.proto"
    "ClassConfidenceThresholding.proto"
    "CustomModel.proto"
    "DictVectorizer.proto"
    "FeatureVectorizer.proto"
    "Gazetteer.proto"
    "GLMClassifier.proto"
    "GLMRegressor.proto"
    "Identity.proto"
    "Imputer.proto"
    "ItemSimilarityRecommender.proto"
    "LinkedModel.proto"
    "NearestNeighbors.proto"
    "NeuralNetwork.proto"
    "NonMaximumSuppression.proto"
    "Normalizer.proto"
    "OneHotEncoder.proto"
    "Scaler.proto"
    "SoundAnalysisPreprocessing.proto"
    "SVM.proto"
    "TextClassifier.proto"
    "TreeEnsemble.proto"
    "VisionFeaturePrint.proto"
    "WordEmbedding.proto"
    "WordTagger.proto"
)

echo "Downloading CoreML proto files from coremltools..."

for proto in "${PROTO_FILES[@]}"; do
    echo "  Downloading $proto..."
    curl -sL "${COREML_BASE_URL}/${proto}" -o "$proto"
done

echo "Adding go_package options..."

# Add go_package for MIL.proto (separate package)
python3 -c "
import re

with open('MIL.proto', 'r') as f:
    content = f.read()

if 'go_package' not in content:
    go_package = 'option go_package = \"github.com/gomlx/go-coreml/proto/coreml/milspec\";'
    content = re.sub(
        r'(option optimize_for[^;]*;)',
        r'\1\n' + go_package,
        content,
        count=1
    )
    with open('MIL.proto', 'w') as f:
        f.write(content)
    print('  Updated MIL.proto')
else:
    print('  MIL.proto already has go_package')
"

# Add go_package for all other protos (shared spec package)
python3 -c "
import os
import re

go_package = 'option go_package = \"github.com/gomlx/go-coreml/proto/coreml/spec\";'

for fname in os.listdir('.'):
    if not fname.endswith('.proto') or fname == 'MIL.proto':
        continue
    with open(fname, 'r') as f:
        content = f.read()
    if 'go_package' not in content:
        content = re.sub(
            r'(option optimize_for[^;]*;)',
            r'\1\n' + go_package,
            content,
            count=1
        )
        with open(fname, 'w') as f:
            f.write(content)
        print(f'  Updated {fname}')
    else:
        print(f'  {fname} already has go_package')
"

echo ""
echo "Creating output directories..."
mkdir -p milspec spec

echo ""
echo "Done! Run 'go generate ./...' from the go-coreml root to regenerate Go code."
echo ""
echo "Proto files downloaded: ${#PROTO_FILES[@]}"
