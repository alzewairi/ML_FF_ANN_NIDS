#! /usr/bin/python
import h2o
import sys
import os
import pandas as pd

cwd = os.getcwdu()
dataset = cwd+"/dataset/unsw-nb15_mereged.zip"
# dataset = cwd+"/dataset/sample.csv"
h2o.connect(ip="localhost", port="54321")
columns_types = ["enum", "numeric", "enum", "numeric", "enum", "enum", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "numeric", "enum", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "time", "time", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "enum", "enum"]
data_frame = h2o.import_file(path=dataset,
                             destination_frame="unsw-nb15_mereged",
                             col_types=columns_types,
                             )
response_column = "label"
training_frame, test_frame, validation_frame = data_frame.split_frame(
    ratios=[0.6, 0.3],
    destination_frames=["unsw-nb15_mereged_training_60",
                        "unsw-nb15_mereged_test_30",
                        "unsw-nb15_mereged_validation_10"
                        ]
)


def dl_modeling(model_id, predictor_columns, runs=10):
    # Set Ignored Columns
    ignored_columns = data_frame.names
    for item in predictor_columns:
        ignored_columns.remove(item)
    ignored_columns.remove(response_column)
    # Run Experiments
    for i in xrange(1, runs+1):
        # Define Model
        model_name = "".join([model_id, "-Run_", str(i)])
        dl_model = h2o.estimators.deeplearning.H2ODeepLearningEstimator(
            model_id=model_name,
            validation_frame=validation_frame,
            ignored_columns=ignored_columns,
            hidden=[10, 10, 10, 10, 10],
            epochs=10,
            nfolds=10,
            activation="rectifier",
            fold_assignment="stratified",
            variable_importances=True,
            single_node_mode=True,
            ignore_const_cols=True,
            score_each_iteration=True,
            shuffle_training_data=True,
            max_w2=float(10),
            l1=1e-5,
        )
        # Train Model
        dl_model.train(
            x=predictor_columns,
            y=response_column,
            training_frame=training_frame,
        )
        # Test Model
        # prediction_results = dl_model.predict(test_frame)
        model_performance = dl_model.model_performance(test_data=test_frame)
        # Output Results
        results_path = cwd+"/results/"+model_name.split('-Run_')[0]
        # Save Model
        h2o.save_model(model=dl_model, path=cwd+"/models", force=True)
        # Export Variable Importance Table
        varimp = dl_model.varimp()
        varimp = pd.DataFrame(varimp)
        varimp.to_csv(
            path_or_buf=results_path+"_VarImp_"+str(i),
            sep="\t",
            na_rep='\t',
            header=('variable', 'relative_importance', 'scaled_importance',	'percentage'),
            index_label='#',
            index=True,
            mode='w',
        )
        # Export Model Output
        original_stdout = sys.stdout
        f = open(cwd+"/output/"+model_name, 'w')
        sys.stdout = f
        print dl_model.show()
        sys.stdout = original_stdout
        f.close()
        # Export Evaluation Metrics
        f = open(results_path+"_TrainingEvalMetrics", 'a+')
        sys.stdout = f
        if i == 1:
            print "Accuracy\tAUC\tF1\tPrecision\tRecall\tSpecificity\tTime"
        print \
            dl_model._model_json['output']['cross_validation_metrics_summary'][1][0], '\t', \
            dl_model._model_json['output']['cross_validation_metrics_summary'][1][1], '\t', \
            dl_model._model_json['output']['cross_validation_metrics_summary'][1][5], '\t', \
            dl_model._model_json['output']['cross_validation_metrics_summary'][1][14], '\t', \
            dl_model._model_json['output']['cross_validation_metrics_summary'][1][16], '\t', \
            dl_model._model_json['output']['cross_validation_metrics_summary'][1][18], '\t', \
            dl_model._model_json['output']['run_time']
        sys.stdout = original_stdout
        f.close()
        # Export Model Performance
        f = open(results_path+"_TestingModelPerformance_"+str(i), 'w')
        sys.stdout = f
        print model_performance
        sys.stdout = original_stdout
        f.close()
        thresholds_and_metric_scores = model_performance['thresholds_and_metric_scores'].as_data_frame()
        thresholds_and_metric_scores = pd.DataFrame(thresholds_and_metric_scores)
        thresholds_and_metric_scores.to_csv(
            path_or_buf=results_path+"_PredictionMetricsThresholds_"+str(i),
            sep="\t",
            na_rep='\t',
            # header=('variable', 'relative_importance', 'scaled_importance',	'percentage'),
            index_label='#',
            index=True,
            mode='w',
        )


def dl_rectifier_5():
    model_name = "dl_Rectifier_ftc_5percent"
    predictor_columns = ["service", "proto", "state", "swin", "sttl", "dttl", "dmeansz", "ct_srv_dst", "dwin", "ct_state_ttl", "trans_depth", "djit", "spkts", "sjit", "ct_dst_sport_ltm", "sloss", "dsport", "sload", "ct_dst_src_ltm"]
    print model_name
    dl_modeling(model_id=model_name, predictor_columns=predictor_columns)


def dl_rectifier_10():
    model_name = "dl_Rectifier_ftc_10percent"
    predictor_columns = ["service", "proto", "state", "swin", "sttl", "dttl", "dmeansz", "ct_srv_dst", "dwin", "ct_state_ttl", "trans_depth", "djit", "spkts", "sjit", "ct_dst_sport_ltm", "sloss", "dsport", "sload", "ct_dst_src_ltm", "ct_srv_src", "dload", "dloss", "synack", "ackdat", "dtcpb"]
    print model_name
    dl_modeling(model_id=model_name, predictor_columns=predictor_columns)


def dl_rectifier_15():
    model_name = "dl_Rectifier_ftc_15percent"
    predictor_columns = ["service", "proto", "state", "swin", "sttl", "dttl", "dmeansz", "ct_srv_dst", "dwin", "ct_state_ttl", "trans_depth", "djit", "spkts", "sjit", "ct_dst_sport_ltm", "sloss", "dsport", "sload", "ct_dst_src_ltm", "ct_srv_src", "dload", "dloss", "synack", "ackdat", "dtcpb", "ct_src_ltm", "tcprtt", "ltime", "stcpb", "smeansz", "dpkts"]
    print model_name
    dl_modeling(model_id=model_name, predictor_columns=predictor_columns)


def dl_rectifier_20():
    model_name = "dl_Rectifier_ftc_20percent"
    predictor_columns = ["service", "proto", "state", "swin", "sttl", "dttl", "dmeansz", "ct_srv_dst", "dwin", "ct_state_ttl", "trans_depth", "djit", "spkts", "sjit", "ct_dst_sport_ltm", "sloss", "dsport", "sload", "ct_dst_src_ltm", "ct_srv_src", "dload", "dloss", "synack", "ackdat", "dtcpb", "ct_src_ltm", "tcprtt", "ltime", "stcpb", "smeansz", "dpkts", "stime", "dur"]
    print model_name
    dl_modeling(model_id=model_name, predictor_columns=predictor_columns)


def dl_rectifier_25():
    model_name = "dl_Rectifier_ftc_25percent"
    predictor_columns = ["service", "proto", "state", "swin", "sttl", "dttl", "dmeansz", "ct_srv_dst", "dwin", "ct_state_ttl", "trans_depth", "djit", "spkts", "sjit", "ct_dst_sport_ltm", "sloss", "dsport", "sload", "ct_dst_src_ltm", "ct_srv_src", "dload", "dloss", "synack", "ackdat", "dtcpb", "ct_src_ltm", "tcprtt", "ltime", "stcpb", "smeansz", "dpkts", "stime", "dur", "sport", "ct_src_dport_ltm"]
    print model_name
    dl_modeling(model_id=model_name, predictor_columns=predictor_columns)


def dl_rectifier_full():
    model_name = "dl_Rectifier_ftc_full"
    predictor_columns = data_frame.names
    predictor_columns.remove(response_column)
    predictor_columns.remove("srcip")
    predictor_columns.remove("dstip")
    predictor_columns.remove("attack_cat")
    print model_name
    dl_modeling(model_id=model_name, predictor_columns=predictor_columns)


def cleanup():
    os.system("rm -rf "+cwd+"/models/*")
    os.system("rm -rf "+cwd+"/output/*")
    os.system("rm -rf "+cwd+"/results/*")

# Perform Tests
cleanup()
dl_rectifier_5()
dl_rectifier_10()
dl_rectifier_15()
dl_rectifier_20()
dl_rectifier_25()
dl_rectifier_full()


h2o.cluster().shutdown(prompt=True)
