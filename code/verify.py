import os
import subprocess
import signal
import time
import math
import sys
from threading import Timer

import tensorflow as tf
import numpy as np
from scriptify import scriptify
from data_preprocessing import get_data
from gloro.models import GloroNet
from autoattack import utils_tf2
from autoattack import AutoAttack
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from marabou_net import MarabouNet, AllowedMisclassifications, Counterexample
from scipy.spatial import distance

import torch

def execute_bash_cmd(cmd, log_file, timeout=3600):
    print("cmd: ", cmd)

    def timerout(p):
        print("Command timed out")
        timer.cancel()
        # os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.kill() # or use p.kill() if shell==False

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    timer = Timer(timeout, timerout, args=[p])
    timer.start()
    # Poll process.stdout to show stdout live
    while True:
        output = p.stdout.readline()
        if p.poll() is not None:
            break
        if output:
            log_file.write(output.decode("utf-8"))
    timer.cancel()

def filter_data(model, X_test, y_test, batch_size, num_classes, isGloro):
    y_pred_correct = []
    y_pred_bot = []
    y_pred_incorrect = []
    eval_samples = X_test.shape[0]
    eval_batch_size = batch_size
    for index in range(eval_samples // eval_batch_size):
        # print(f'Evaluating model: Iteration {index} of {eval_samples // eval_batch_size}')
        strt_indx = index * eval_batch_size
        end_indx = (index + 1) * eval_batch_size if (index + 1 < (eval_samples // eval_batch_size)) \
            else eval_samples
        x, y = X_test[strt_indx:end_indx, ], y_test[strt_indx:end_indx, ]
        y = np.argmax(y, axis=-1)
        y_pred = np.argmax(model.predict(x), axis=-1)
        y_pred_correct_t = (y_pred == y)
        y_pred_incorrect_t = np.logical_not(y_pred_correct_t)
        y_pred_correct.append(y_pred_correct_t)
        y_pred_incorrect.append(y_pred_incorrect_t)

        if isGloro:
            y_pred_bot_t = (y_pred == num_classes)
            y_pred_bot.append(y_pred_bot_t)


    y_pred_correct = 1.0 * np.concatenate(y_pred_correct)
    y_pred_incorrect = 1.0 * np.concatenate(y_pred_incorrect)
    correct_idx = np.flatnonzero(y_pred_correct)
    incorrect_idx = np.flatnonzero(y_pred_incorrect)
    print("Num of samples handled correctly: ", np.shape(correct_idx))
    print("Num of samples handled incorrectly: ", np.shape(incorrect_idx))

    if isGloro:
        y_pred_bot = 1.0 * np.concatenate(y_pred_bot)
        bot_idx = np.flatnonzero(y_pred_bot)
        print("Num of samples handled bot: ", np.shape(bot_idx))
        x_filtered = np.squeeze(np.take(X_test, bot_idx, axis=0))
        y_filtered = np.squeeze(np.take(y_test, bot_idx))
    else:
        x_filtered = np.squeeze(np.take(X_test, correct_idx, axis=0))
        y_filtered = np.squeeze(np.take(y_test, correct_idx))

    print("x_filtered shape: ", np.shape(x_filtered))
    print("y_filtered shape: ", np.shape(y_filtered))

    return x_filtered, y_filtered

def print_marabou_query(x, y, epsilon, path, num_classes, isOver=False):
    labels = list(range(num_classes))
    incorrect_labels = list(filter(lambda l: l != y, labels))
    names = []
    for i in range(ord('a'), ord('a') + num_classes - 1):
        names.append(chr(i))

    if isOver:
        dist = epsilon
    else:
        dist = epsilon/math.sqrt(2)

    for i in range(0, len(incorrect_labels)):
        fpath = f'{path}{names[i]}.txt'
        with open(fpath, 'w') as f1:
            for j in range(np.shape(x)[0]):
                x_lb = x[j] - dist
                x_ub = x[j] + dist
                f1.write(f'x{j} >= {x_lb}\n')
                f1.write(f'x{j} <= {x_ub}\n')
            f1.write(f'+y{incorrect_labels[i]} -y{y} >= 0')
    return names

def analyze_marabou_log(logpath):
    out_term = subprocess.check_output(["tail", "-n", "1",
                                        logpath])
    out_term = out_term.decode("utf-8")

    p1 = subprocess.Popen(["grep", "x0 = ", logpath],
                          stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["wc", "-l"], stdin=p1.stdout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1.stdout.close()  # Allow proc1 to receive a SIGPIPE if proc2 exits.
    out, err = p2.communicate()
    if not out_term.startswith('unsat'):
        print("sat found")
        num_viol = int(out.decode("utf-8"))
        return num_viol
    return 0

def analyze_marabou_log_for_soln(logpath, num_features):
    soln = []
    for i in range(num_features):
        out_term = subprocess.check_output(["grep", f'x{i} = ', logpath])
        out_term = out_term.decode("utf-8")
        soln.append(float(out_term))
    soln = np.array(soln, dtype=np.float32)
    return soln

def execute_marabou_query(x, y, epsilon, nnet, isOver=False):
    if isOver:
        dist = epsilon
    else:
        dist = epsilon/math.sqrt(2)
    lbs = x - dist
    ubs = x + dist
    return nnet.find_counterexample(lbs, ubs, y)


if __name__ == '__main__':

    @scriptify
    def script(experiment='safescad',
               batch_size=128,
               dataset_file=None,
               conf_name='default',
               epsilon=0.5,
               marabou_path=None,
               use_marabou_api=False,
               attack='cleverhans', # or 'autoattack'
               gpu=0):

        print("Configuration Options:")
        print("experiment=", experiment)
        print("batch_size=", batch_size)
        print("dataset_file=", dataset_file)
        print("conf_name=", conf_name)
        print("epsilon=", epsilon)
        print("marabou_path=", marabou_path)
        print("use_marabou_api=", use_marabou_api)
        print("attack=", attack)

        data_dir = f'./experiments/data/{experiment}/{conf_name}'
        model_dir = f'./experiments/models/{experiment}/{conf_name}'
        query_dir = f'./experiments/marabou-queries/{experiment}/{conf_name}'
        log_dir = f'./experiments/marabou-queries/{experiment}/{conf_name}/logs'
        summary_dir = f'./experiments/marabou-queries/{experiment}/{conf_name}/summaries'

        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        # LOAD DATA
        X_train, y_train, X_test, y_test = get_data(experiment, dataset_file, data_dir)
        num_features = X_train.shape[1]
        num_classes = y_train.shape[1]
        num_samples = y_test.shape[0]

        # Load model
        model = tf.keras.models.load_model(f'{model_dir}/model.h5')
        gloro_model = GloroNet(model=model, epsilon=epsilon)

        # Get test samples where model is not robust
        print("Certifying model using Gloro Lipschitzness check ...")
        x_filtered, y_filtered = filter_data(gloro_model, X_test, y_test, batch_size, num_classes, True)
        num_samples_postGloro = y_filtered.shape[0]

        # Attack test samples where model is not robust via AutoAttack
        if attack == 'autoattack':
            print("Attacking model using AutoAttack ...")
            model_adapted = utils_tf2.ModelAdapter(model)
            adversary = AutoAttack(model_adapted, norm='L2', eps=epsilon, version='standard', is_tf_model=True)
            torch_x_filtered = torch.from_numpy(np.transpose(x_filtered, (0, 3, 1, 2))).float().cuda()
            torch_y_filtered = torch.from_numpy(y_filtered).float()
            X_adv = adversary.run_standard_evaluation(torch_x_filtered, torch_y_filtered, bs=batch_size)
            X_adv = np.moveaxis(X_adv.cpu().numpy(), 1, 3)

            x_filtered2, y_filtered2 = filter_data(model, X_adv, y_test, batch_size, num_classes, False)
        elif attack == 'cleverhans':
            print("Attacking model using Cleverhans ...")
            X_adv = []
            eval_samples = x_filtered.shape[0]
            eval_batch_size = batch_size
            for index in range(eval_samples // eval_batch_size):
                #print(f'Evaluating model: Iteration {index} of {eval_samples // eval_batch_size}')
                strt_indx = index * eval_batch_size
                end_indx = (index + 1) * eval_batch_size if (index + 1 < (eval_samples // eval_batch_size)) \
                    else eval_samples
                x, y = x_filtered[strt_indx:end_indx, ], y_filtered[strt_indx:end_indx, ]
                x_adv = projected_gradient_descent(model, x, epsilon, 0.01, 40, 2)
                X_adv.append(x_adv)

            X_adv = np.concatenate(X_adv)
            x_filtered2, y_filtered2 = filter_data(model, X_adv, y_test, batch_size, num_classes, False)
            num_samples_postattack = y_filtered2.shape[0]

        # Verify test samples where model is not robust via Marabou
        if marabou_path is not None and not use_marabou_api:
            print("Certifying model using Marabou ...")
            x_cex_idxs = []
            x_pf_indxs = []
            for index in range(x_filtered2.shape[0]):
                print(f'Verifying model: Iteration {index} of {x_filtered2.shape[0]}')
                x, y = x_filtered2[index, ], y_filtered2[index, ]
                qnames = print_marabou_query(x, y, epsilon, f'{query_dir}/query_{index}_under_', num_classes, isOver=False)

                marabou_found_cex = False
                for qname in qnames:
                    # cmd_str = [marabou_path, "--input", model_dir + "/model.nnet",
                    #                            "--property", query_dir + "/query_" + index + "_under_" + qname + ".txt",
                    #                            "--summary-file", summary_dir + "/query_" + index + "_under_" + qname + ".txt",
                    #            "--verbosity", "1", "--snc", "--split-strategy", "polarity",
                    #            "--num-workers", "6", "--initial-divides", "4", "--initial-timeout", "0"]

                    cmd_str = [marabou_path, "--input", model_dir + "/model.nnet",
                               "--property", query_dir + "/query_" + str(index) + "_under_" + qname + ".txt",
                               "--summary-file", summary_dir + "/query_" + str(index) + "_under_" + qname + ".txt",
                               "--verbosity", "1"]

                    log_file = open(f'{log_dir}/query_{index}_under_{qname}.log', "w")
                    execute_bash_cmd(cmd_str, log_file)
                    log_file.close()

                    num_cex = analyze_marabou_log(f'{log_dir}/query_{index}_under_{qname}.log')
                    if num_cex != 0:
                        cex = analyze_marabou_log_for_soln(f'{log_dir}/query_{index}_under_{qname}.log', num_features)
                        y_pred = np.argmax(model.predict(np.expand_dims(cex, axis=0)))
                        dist_cex = distance.euclidean(x, cex)
                        if y_pred != y and dist_cex <= epsilon:
                            print('Found valid underapproximate counterexample')
                            marabou_found_cex = True
                            x_cex_idxs.append(index)
                            break
                        else:
                            print('Found invalid underapproximate counterexample')

                marabou_found_proof = True
                if not marabou_found_cex:
                    qnames = print_marabou_query(x, y, epsilon, f'{query_dir}/query_{index}_over_', num_classes, isOver=True)

                    for qname in qnames:
                        # cmd_str = [marabou_path, "--input", model_dir + "/model.nnet",
                        #                                "--property", query_dir + "/query_" + index + "_over_" + qname + ".txt",
                        #                                "--summary-file", summary_dir + "/query_" + index + "_over_" + qname + ".txt",
                        #            "--verbosity", "1", "--snc", "--split-strategy", "polarity",
                        #            "--num-workers", "6", "--initial-divides", "4", "--initial-timeout", "0"]

                        cmd_str = [marabou_path, "--input", model_dir + "/model.nnet",
                                   "--property", query_dir + "/query_" + str(index) + "_over_" + qname + ".txt",
                                   "--summary-file", summary_dir + "/query_" + str(index) + "_over_" + qname + ".txt",
                                   "--verbosity", "1"]

                        log_file = open(f'{log_dir}/query_{index}_over_{qname}.log', "w")
                        execute_bash_cmd(cmd_str, log_file)
                        log_file.close()

                        num_cex = analyze_marabou_log(f'{log_dir}/query_{index}_over_{qname}.log')
                        if num_cex != 0:
                            cex = analyze_marabou_log_for_soln(f'{log_dir}/query_{index}_over_{qname}.log',
                                                               num_features)
                            y_pred = np.argmax(model.predict(np.expand_dims(cex, axis=0)))
                            dist_cex = distance.euclidean(x, cex)
                            if y_pred != y and dist_cex <= epsilon:
                                print('Found valid overapproximate counterexample')
                            else:
                                print('Found invalid overapproximate counterexample')
                            marabou_found_proof = False
                            break

                    if marabou_found_proof:
                        print('Found proof')
                        x_pf_indxs.append(index)
            print("Num of samples total: ", num_samples)
            print("Num of samples certified bot by Gloro: ", num_samples_postGloro)
            print("Num of samples not successfully attacked: ", num_samples_postattack)
            print("Num of samples with Marabou counterexamples: ", len(x_cex_idxs))
            print("Num of samples certified robust by Marabou: ", len(x_pf_indxs))
            print("Num of samples unresolved samples: ", num_samples_postattack - len(x_cex_idxs) - len(x_pf_indxs))

        elif marabou_path is not None and use_marabou_api:
            sys.path.append(os.path.abspath(marabou_path))
            print("Certifying model using Marabou ...")
            marabou_net = MarabouNet(f'{model_dir}/model.nnet', network_options=dict(), marabou_options=dict(),
                                     marabou_verbosity=1)
            x_cex_idxs = []
            x_pf_indxs = []
            for index in range(x_filtered2.shape[0]):
                print(f'Verifying model: Iteration {index} of {x_filtered2.shape[0]}')
                x, y = x_filtered2[index,], y_filtered2[index,]

                m_pred, cex = execute_marabou_query(x, y, epsilon, marabou_net, isOver=False)
                marabou_found_cex = False
                if cex is not None:
                    y_pred = np.argmax(model.predict(np.expand_dims(cex, axis=0)))
                    dist_cex = distance.euclidean(x, cex)
                    if y_pred != y and dist_cex <= epsilon:
                        print('Found valid underapproximate counterexample')
                        marabou_found_cex = True
                        x_cex_idxs.append(index)
                    else:
                        print('Found invalid underapproximate counterexample')

                marabou_found_proof = True
                if not marabou_found_cex:
                    m_pred, cex = execute_marabou_query(x, y, epsilon, marabou_net, isOver=True)
                    if cex is not None:
                        marabou_found_proof = False
                        y_pred = np.argmax(model.predict(np.expand_dims(cex, axis=0)))
                        dist_cex = distance.euclidean(x, cex)
                        if y_pred != y and dist_cex <= epsilon:
                            print('Found valid overapproximate counterexample')
                        else:
                            print('Found invalid overapproximate counterexample')
                    else:
                        print('Found proof')
                        x_pf_indxs.append(index)

            print("Num of samples total: ", num_samples)
            print("Num of samples certified bot by Gloro: ", num_samples_postGloro)
            print("Num of samples not successfully attacked: ", num_samples_postattack)
            print("Num of samples with Marabou counterexamples: ", len(x_cex_idxs))
            print("Num of samples certified robust by Marabou: ", len(x_pf_indxs))
            print("Num of samples unresolved samples: ", num_samples_postattack - len(x_cex_idxs) - len(x_pf_indxs))