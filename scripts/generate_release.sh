#!/bin/bash
rm releases/$1 2> /dev/null # remove if it exists

zip -r releases/$1 \
        audicor_reader \
        data \
        evaluation \
        loss \
        model \
        releases \
        scripts \
        train \
        utils \
        config.cfg \
        Pipfile* \
        readme.md \
        test.py \
        train.py \
        tutorial.ipynb \
        -x '*.git*' \
        -x '*__pycache__*' \
        -x '*tkdnd*' \
        -x 'data/*big_exam*' \
        -x 'data/*IEC*' \
        -x 'data/*.pt*' \
        -x 'data/*.xls*' \
        -x 'data/*.json' \
        -x '*.ipynb_checkpoints*' \
        -x 'visualization.ipynb' \