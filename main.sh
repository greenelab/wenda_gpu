#!/bin/bash

# This is designed to be a one-click way to run wenda. Be sure to check all parameters in the "Parameters to Edit" section before running. Users who want more control (e.g. for scheduling jobs on a cluster) can use this script as a guide; run train_feature_models.py in small batches until all feature models have been trained, and then run train_elastic_net.py.


### PARAMETERS TO EDIT
# DATASET-SPECIFIC INFORMATION. Change prefix to desired dataset identifier and confirm all other settings are as desired
prefix="original" #This should be changed to a specific, meaningful identifier for the dataset and classification problem.
batch_size=100 #How many feature models to train in one batch. For small datasets (i.e. few number of samples), this number can be raised without risk of memory flow errors, but very large datasets may require a number <100.
horvath=0  #Transformation used for age prediction tasks. This value should be 0 except in this specific use case, when it should be 1.
logistic=0 #This value should be 1 for logistic net regression (binary label data) and 0 for elastic net regression (continuous label data).
delimiter="\t" #Field separator for input and output files.

# PATH INFORMATION. Paths are relative to the wenda_gpu root directory. These parameters should only need to be modified if the user wants to use data from or store output in another location.
data_path="data" #Location of input data (i.e. a single repository with source_data.tsv, target_data.tsv, and source_y.tsv)
feature_model_path="feature_models" #Where feature model .pth files will be written to
confidence_path="confidences" #Where confidence scores for each feature model will be written to
elastic_net_path="output" #Where model predictions will be written to

### END PARAMETERS TO EDIT


### CODE
cd "$(dirname "$0")"

mkdir -p $data_path
mkdir -p $feature_model_path
mkdir -p $confidence_path
mkdir -p $elastic_net_path

# Get number of columns
source_features=`awk -F"\t" '{print NF;exit}' $data_path/$prefix/source_data.tsv`
target_features=`awk -F"\t" '{print NF;exit}' $data_path/$prefix/target_data.tsv`
if [ $source_features -ne $target_features ]; then
	echo "Error: Source and target datasets have different numbers of features."
	echo "Source dataset has ${source_features} features, target dataset has ${target_features}."
	echo "Confirm your datasets are laid out so samples are rows and features are columns."
	exit 1
else
	echo "Preparing to train models for ${source_features} features..."
fi

# Calculate number of batches to run
batches=$(( $source_features / $batch_size ))
if [ $(( $source_features % $batch_size)) != "0" ]; then
	batches=$((batches + 1))
fi

# Train feature models in batches
for (( i=0; i<$batches; i++))
do
	start=$(( $i * $batch_size ))
	stop=$(( $start + $batch_size - 1 ))
	echo "Training models ${start} to ${stop}..."
	python3 train_feature_models.py -p ${prefix} -s ${start} -r ${batch_size} -d ${delimiter} \
	       	--data_path ${data_path} --feature_model_path ${feature_model_path} \
		--confidence_path ${confidence_path}
done

# Confirm all feature models have been trained and confidence scores generated
conf_files=`ls -1q $confidence_path/$prefix | wc -l`
echo $conf_files

if [ $conf_files -ne $source_features ]; then
	echo "Error: not all models trained. If the script timed out, rerunning main.sh"
	echo "should train the missing models without overwriting the existing ones."
	echo "If there was a memory overflow error, try setting the 'batch_size' parameter"
	echo "to a smaller value and rerunning main.sh."
	echo "If the problem persists, try running python3 train_feature_models.py on a"
	echo "range that includes a model missing from confidences folder and check for errors."
	exit 1
fi

# Once all feature models are trained, run elastic net
if [ horvath ]; then
	python3 train_elastic_net.py -p ${prefix} -d ${delimiter} --data_path ${data_path} \
	       --confidence_path ${confidence_path} --elastic_net_path ${elastic_net_path} --horvath
elif [ logistic ]; then
	python3 train_elastic_net.py -p ${prefix} -d ${delimiter} --data_path ${data_path} \
		--confidence_path ${confidence_path} --elastic_net_path ${elastic_net_path} --logistic
else
	python3 train_elastic_net.py -p ${prefix} -d ${delimiter} --data_path ${data_path} \
		--confidence_path ${confidence_path} --elastic_net_path ${elastic_net_path}
fi

end=$SECONDS
echo "Wenda complete. Time to run (seconds): $end"
