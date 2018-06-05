#! /usr/bin/python
import h2o
import sys
import os
import pandas as pd

cwd = os.getcwdu()
dataset = cwd+"/dataset/unsw-nb15_mereged.zip"
h2o.connect(ip="localhost", port="54321")
columns_types = ["enum", "numeric", "enum", "numeric", "enum", "enum", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "numeric", "enum", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "time", "time", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                 "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "enum", "enum"]
data = h2o.import_file(path=dataset,
                       destination_frame="unsw-nb15_mereged",
                       col_types=columns_types,
                       )
response_column = "label"
predictor_columns = data.names
predictor_columns.remove(response_column)
predictor_columns.remove("srcip")
predictor_columns.remove("dstip")
predictor_columns.remove("attack_cat")
# training, test = data.split_frame(
#     ratios=[0.7],
#     destination_frames=["unsw-nb15_mereged_70", "unsw-nb15_mereged_30"]
# )


def dl_modeling(model_id, activation_function, runs=10):
    for i in xrange(1, runs+1):
        model_name = "".join([model_id, "-Run_", str(i)])
        dl_model = h2o.estimators.deeplearning.H2ODeepLearningEstimator(
            model_id=model_name,
            validation_frame=data,
            ignored_columns=["srcip", "dstip", "attack_cat"],
            hidden=[10, 10, 10, 10, 10],
            epochs=10,
            nfolds=10,
            activation=activation_function,
            fold_assignment="stratified",
            variable_importances=True,
            single_node_mode=True,
            ignore_const_cols=True,
            score_each_iteration=True,
            shuffle_training_data=True,
            max_w2=float(10),
            l1=1e-5,
        )

        dl_model.train(
            x=predictor_columns,
            y=response_column,
            training_frame=data,
        )

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
        f = open(results_path+"_EvalMetrics", 'a+')
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


def dl_tanh():
    model_name = "dl_Tanh_aft"
    print model_name
    dl_modeling(model_id=model_name, activation_function="tanh")


def dl_tanh_dropout():
    model_name = "dl_TahnWithDropout_aft"
    print model_name
    dl_modeling(model_id=model_name, activation_function="tanhwithdropout")


def dl_rectifier():
    model_name = "dl_Rectifier_aft"
    print model_name
    dl_modeling(model_id=model_name, activation_function="rectifier",)


def dl_rectifier_dropout():
    model_name = "dl_RectifierWithDropout_aft"
    print model_name
    dl_modeling(model_id=model_name, activation_function="rectifierwithdropout")


def dl_maxout():
    model_name = "dl_Maxout_aft"
    print model_name
    dl_modeling(model_id=model_name, activation_function="maxout")


def dl_maxout_dropout():
    model_name = "dl_MaxoutWithDropout_aft"
    print model_name
    dl_modeling(model_id=model_name, activation_function="maxoutwithdropout")


def cleanup():
    os.system("rm -rf "+cwd+"/models/*")
    os.system("rm -rf "+cwd+"/output/*")
    os.system("rm -rf "+cwd+"/results/*")


# Perform Tests
# cleanup()
dl_tanh()
dl_tanh_dropout()
dl_rectifier()
dl_rectifier_dropout()
dl_maxout()
dl_maxout_dropout()


h2o.cluster().shutdown(prompt=True)
