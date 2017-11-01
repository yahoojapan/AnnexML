#!/bin/bash

function download_from_google_drive() {
    COOKIE_FILE=$(mktemp)
    CONFIRM_ID=$(curl -c $COOKIE_FILE -s -L "https://drive.google.com/uc?export=download&id=$2" | grep confirm |  sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/")
    curl -b $COOKIE_FILE -L -o $1 "https://drive.google.com/uc?confirm=${CONFIRM_ID}&export=download&id=$2"
    rm $COOKIE_FILE

    return 0
}


function train_and_evaluate() {
    TARGET=$1
    GDRIVE_ID=$2

    TARGET_DATA_DIR=$DATA_DIR/$TARGET

    if [ ! -d $TARGET_DATA_DIR ]; then
        pushd $DATA_DIR
        if [ ! -f ${TARGET}.zip ]; then
            download_from_google_drive ${TARGET}.zip $GDRIVE_ID
        fi
        unzip ${TARGET}.zip
        popd
    fi

    FILE_HEAD=$(echo $TARGET | sed -e 's/^\(.\)/\L\1/')
    TRAIN_FILE=$TARGET_DATA_DIR/${FILE_HEAD}_train.txt
    PREDICT_FILE=$TARGET_DATA_DIR/${FILE_HEAD}_test.txt
    MODEL_FILE=$WORK_DIR/annexml-model-${TARGET}.bin
    RESULT_FILE=$WORK_DIR/annexml-result-${TARGET}.txt


    echo "----------------------------------------"
    echo $TARGET

    $SRC_DIR/annexml train annexml-example.json train_file=${TRAIN_FILE} model_file=${MODEL_FILE}
    $SRC_DIR/annexml predict annexml-example.json predict_file=${PREDICT_FILE} model_file=${MODEL_FILE} result_file=${RESULT_FILE}

    cat ${RESULT_FILE} | python $SCRIPTS_DIR/learning-evaluate_predictions.py
    cat ${RESULT_FILE} | python $SCRIPTS_DIR/learning-evaluate_predictions_propensity_scored.py $TRAIN_FILE -A $3 -B $4

    echo "----------------------------------------"
    echo ""

    return 0
}



cd $(dirname $0)

SRC_DIR=$(cd src && pwd)
SCRIPTS_DIR=$(cd scripts && pwd)

if [ ! -d data ]; then
    mkdir data
fi
DATA_DIR=$(cd data && pwd)

if [ ! -d work ]; then
    mkdir work
fi
WORK_DIR=$(cd work && pwd)


if [ ! -x $SRC_DIR/annexml ]; then
    make -C $SRC_DIR annexml
fi


#train_and_evaluate AmazonCat "0B3lPMIHmG6vGa2tMbVJGdDNSMGc" 0.55 1.5
train_and_evaluate Wiki10 "0B3lPMIHmG6vGaDdOeGliWF9EOTA" 0.55 1.5
#train_and_evaluate DeliciousLarge "0B3lPMIHmG6vGR3lBWWYyVlhDLWM" 0.55 1.5
#train_and_evaluate WikiLSHTC "0B3lPMIHmG6vGSHE1SWx4TVRva3c" 0.5 0.4
#train_and_evaluate Amazon "0B3lPMIHmG6vGdUJwRzltS1dvUVk" 0.6 2.6

