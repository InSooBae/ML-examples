@echo off
REM ===========================================================================================
REM  TRAINING_ID, Param1, Param2 설정할 것
REM ===========================================================================================

Set TRAINING_ID=SD_TEST

Set Param1= ^
--wanted_words=yes,no,up,down,left,right,on,off,stop,go ^
--model_architecture dnn ^
--model_size_info 436 436 436  ^
--dct_coefficient_count 10  ^
--window_size_ms 40 --window_stride_ms 40 ^
--clip_duration_ms 1000 ^
--dct_coefficient_count 10


Set Param2= ^
--unknown_percentage=100 ^
--how_many_training_steps 10000,10000,10000 ^
--learning_rate 0.005,0.0001,0.00002 ^
--data_url= --data_dir=/tmp/speech_dataset ^
--eval_step_interval=400 --save_step_interval=100 ^
--testing_percentage 7 ^
--validation_percentage 7 ^
--background_volume 0.3 ^
--background_frequency 0.8 ^
--batch_size 100

Set Param_Retraining= ^
--unknown_percentage=100 ^
--how_many_training_steps 10000^
--learning_rate 0.00002 ^
--data_url= --data_dir=/tmp/speech_dataset ^
--eval_step_interval=400 --save_step_interval=100 ^
--testing_percentage 7 ^
--validation_percentage 7 ^
--background_volume 0.3 ^
--background_frequency 0.8 ^
--batch_size 100

REM ===========================================================================================
REM  여기부터는 신경쓰지 않아도 됨.
REM ===========================================================================================

Set TRAINING_DIR=work\DNN\%TRAINING_ID%
Set MODEL_EXP_DIR=work\DNN\%TRAINING_ID%\Exp_model

if "%1"=="freeze" goto MODEL_FREEZE
if "%1"=="retrain" goto RETRAIN

@echo %date% %time%


:TRAINING
REM ===========================================================================================
REM // Training 

if not exist %TRAINING_DIR% mkdir %TRAINING_DIR%

@echo on
python train.py %Param1% %Param2% --summaries_dir %TRAINING_DIR%\retrain_logs --train_dir %TRAINING_DIR%\training 
@echo off
echo %date% %time% >> %TRAINING_DIR%\log.txt
echo python train.py %Param1% %Param2% --summaries_dir %TRAINING_DIR%\retrain_logs --train_dir %TRAINING_DIR%\training >> %TRAINING_DIR%\log.txt
goto MODEL_FREEZE


:RETRAIN
REM ===========================================================================================
REM // ReTraining

setlocal
set start_checkpoint=

:WAIT_START_CHECKPOINT
set /p start_checkpoint=Start Check Point 이름을 입력하세요(ex : dnn_9722.ckpt-45000):
if "%start_checkpoint%" == "" goto WAIT_START_CHECKPOINT

@echo on
python train.py %Param1% %Param_Retraining% --start_checkpoint %TRAINING_DIR%\training\best\%start_checkpoint% --summaries_dir %TRAINING_DIR%\retrain_logs --train_dir %TRAINING_DIR%\training 
@echo off
echo %date% %time% >> %TRAINING_DIR%\log.txt
echo python train.py %Param1% %Param_Retraining% --start_checkpoint %TRAINING_DIR%\training\best\%start_checkpoint% --summaries_dir %TRAINING_DIR%\retrain_logs --train_dir %TRAINING_DIR%\training >> %TRAINING_DIR%\log.txt



:MODEL_FREEZE
REM ===========================================================================================
REM // Freeze Model (freeze to "best_dnn.pb" )

if not exist %MODEL_EXP_DIR% mkdir %MODEL_EXP_DIR%


setlocal
set bestmodel_name=

:WAIT_BEST_MODEL_NAME
set /p bestmodel_name=베스트 모델 이름을 입력하세요(ex : dnn_9722.ckpt-45000):
if "%bestmodel_name%" == "" goto WAIT_BEST_MODEL_NAME

copy %TRAINING_DIR%\training\best\%bestmodel_name%.* %MODEL_EXP_DIR%\.

@echo on
python get_weight.py %Param1% --checkpoint %MODEL_EXP_DIR%\%bestmodel_name% --output_file %MODEL_EXP_DIR%\weights-best_dnn.txt
@echo off

echo %date% %time% >> %TRAINING_DIR%\log.txt
echo python get_weight.py %Param1% --checkpoint %MODEL_EXP_DIR%\%bestmodel_name% --output_file %MODEL_EXP_DIR%\weights-best_dnn.txt >> %TRAINING_DIR%\log.txt





:END
