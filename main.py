import numpy as np
import sys
import os
import csv
from GMM import GMM
from scipy import linalg
import matplotlib as m
m.use('Agg')
import matplotlib.pyplot as plt

def data_loading(file_name):
    temp_data= []
    # read data from a file
    with open(file_name, 'r') as f:
        data = f.readlines()
        temp_data = [line.split() for line in data]
    # convert 2d string list to 2d float numpy array
    final_data = np.zeros((len(temp_data), 2))
    for row_ind in range(len(temp_data)):
        final_data[row_ind, 0] = float(temp_data[row_ind][0])
        final_data[row_ind, 1] = float(temp_data[row_ind][1])
    return final_data

def save_data_csv_file(input_data):
    np.savetxt('GMM_dataset.csv', input_data, delimiter=",")

def label_assignment(clusters, in_data):
    data_label = np.zeros(in_data.shape[0])
    cov_collec = []
    for data_ind in range(in_data.shape[0]):
        # compute the subtraction
        subtraction = in_data[data_ind, :] - clusters
        # compute L2 norm of each cluster and pick up the minimum value
        l2_norm_list = np.zeros(clusters.shape[0])
        for K_ind in range(clusters.shape[0]):
            l2_norm_list[K_ind] = subtraction[K_ind, :].dot(subtraction[K_ind, :])
        # =============================================================
        data_label[data_ind] = np.argmin(l2_norm_list)
    
    # create a new data set with data labels
    data_label_col = data_label.reshape((-1, 1))
    temp_data = np.concatenate((in_data, data_label_col), axis=1)
    # sort the data set based on labels
    temp_data = temp_data[temp_data[:, 2].argsort()]
    # compute covariances
    start_point = 0
    for cluster_ind in range(clusters.shape[0]):
        if cluster_ind == 0:
            temp_matrix = temp_data[0:list(data_label).count(cluster_ind), 0:2]
            cov_collec.append(np.cov(temp_matrix.T))
        else:
            start_point += list(data_label).count(cluster_ind - 1)
            end_point = start_point + list(data_label).count(cluster_ind)
            temp_matrix = temp_data[start_point:end_point, 0:2]
            cov_collec.append(np.cov(temp_matrix.T))
    return data_label, cov_collec

def label_GMM(in_prediction):
    data_label = np.zeros(in_prediction.shape[0])
    for index in range(in_prediction.shape[0]):
        data_label[index] = np.argmin(in_prediction[index, :])
    return data_label

def find_the_best(in_likelihood, in_parameter):
    # find the best log-likelihood
    temp = []
    for element in in_likelihood:
        temp.append(element[-1])
    max_index = temp.index(max(temp))
    return in_likelihood[max_index],in_parameter['mu'][max_index], in_parameter['cov'][max_index], in_parameter['prior'][max_index]

def drawing_Gaussian(mu, s, training_data, best_mu, flag, in_ind):
    gmm_param = {}
    # plot the figure
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig, ax = plt.subplots()
    ax.scatter(training_data[:,0], training_data[:,1], c='b', alpha=0.5)
    plt.xlabel('Feature: x1')
    plt.ylabel('Feature: x2')
    if flag == 1:
        plt.title('Gaussian Mixture Model (Trained Model)')
        ax.scatter(mu[:, 0], mu[:, 1], c='r', s = 100, alpha = 0.5)
    elif flag == 2:
        plt.title('Gaussian Mixture Model (True Model)')
        ax.scatter(best_mu[:, 0], best_mu[:, 1], c='r', s = 100, alpha = 0.5)
    # create ellipses
    color = ['y', 'gold', 'cornflowerblue', 'c', 'darkorange']
    for index in range(mu.shape[0]):
        temp_list = []
        temp_list.append(mu[index, :])
        temp_list.append(s[index])
        gmm_param[str(index)] = temp_list
        v, w = linalg.eigh(s[index])
        v = 3.5 * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = m.patches.Ellipse(mu[index, :], v[0], v[1], 180. + angle, color=color[index])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    if flag == 1:
        fig.savefig('./gmm_result/iter' + str(in_ind) + '.png')
    elif flag == 2:
        fig.savefig('true_gaussian_function.png')
    return gmm_param

def drawing_Log_likelihood(in_list):
    # plot the figure
    plt.rcParams.update({'figure.max_open_warning': 0})
    new_list = in_list[1:]
    x = list(range(len(new_list)))
    fig, ax = plt.subplots()
    ax.plot(x, new_list)
    plt.xlabel('Epoch')
    plt.ylabel('Log-likelihood')
    plt.title('Log-likelihood Overall All Epoch')
    fig.savefig('log-likelihood.png')
    
def main():
    # declare variables
    K_value = 3
    max_epoch = 1000
    repeat_num = 8
    repeat_num_gmm = 1
    name = 'GMM_dataset.txt'

    # get the training data
    training_data = data_loading(name)
        
    if (sys.argv[1] == 'kmean'):
        # run the K-mean program
        program_name = './a.out'
        parameter_line = ' ' + 'training_kmeans ' + str(K_value) + ' ' + str(repeat_num) + ' 0'
        print('Running K-mean')
        os.system(program_name + parameter_line)
        print('The Program is done.')

        # read minimum SSE Position
        min_sse_pos = np.loadtxt('./min_sse_pos.csv',delimiter=',',skiprows=0)
        
        # read clusters from K-mean algorithm
        kmean_name = './all_cluster_center' + str(int(min_sse_pos)) + '.csv'
        kmean_clusters = np.loadtxt(kmean_name,delimiter=',',skiprows=0)
        
        for draw_ind in range(int(kmean_clusters.shape[0] / K_value)):
            # the last index: (int(kmean_clusters.shape[0] / K_value) - 1)
            start_index = draw_ind * K_value
            iter_kmean_clusters = kmean_clusters[start_index:start_index + K_value, :]

            # label assignment
            out_label, cov_list = label_assignment(iter_kmean_clusters, training_data)

            # save the figures
            plt.rcParams.update({'figure.max_open_warning': 0})
            fig, ax = plt.subplots()
            ax.scatter(training_data[:,0], training_data[:,1], c=out_label, alpha=0.5)
            ax.scatter(iter_kmean_clusters[:, 0], iter_kmean_clusters[:, 1], c='b', s = 100, alpha = 0.5)
            plt.xlabel('Feature: x1')
            plt.ylabel('Feature: x2')
            plt.title('K-mean Clustering')
            fig.savefig('./kmean_result/iter' + str(draw_ind) + '.png')
            #fig.clf()            
    elif (sys.argv[1] == 'gmm'):
        # read minimum SSE Position
        min_sse_pos = np.loadtxt('./min_sse_pos.csv',delimiter=',',skiprows=0)
        
        # read clusters from K-mean algorithm
        kmean_name = './all_cluster_center' + str(int(min_sse_pos)) + '.csv'
        kmean_clusters = np.loadtxt(kmean_name,delimiter=',',skiprows=0)
        start_index = (int(kmean_clusters.shape[0] / K_value) - 1) * K_value
        iter_kmean_clusters = kmean_clusters[start_index:start_index + K_value, :]
        out_label, cov_list = label_assignment(iter_kmean_clusters, training_data)
        
        # call GMM class
        gmm = GMM(K_value, repeat_num_gmm, max_epoch, training_data)
        all_likelihood, parameters = gmm.model_training(iter_kmean_clusters, out_label, cov_list)
        true_mu, true_covariance = gmm.fit_true_model(training_data)
        # find the mu, covariance, and prior
        best_likelihood, all_mu, all_cov, all_prior = find_the_best(all_likelihood, parameters)
        
        # prediction phase
        prediction = gmm.model_predict(all_mu[-1], all_cov[-1], all_prior[-1], training_data)
        labels = label_GMM(prediction)
        
        # drawing gaussian functions
        for ind in range(len(all_mu)):
            out_para = drawing_Gaussian(all_mu[ind], all_cov[ind], training_data, all_mu[-1], 1, ind)
            if ind == (len(all_mu) - 1):
                # print out parameters of the Gaussian function
                for ind_2 in range(K_value):
                    print('Cluster:', ind_2)
                    print('Mu:')
                    print(out_para[str(ind_2)][0])
                    print('Covariance:')
                    print(out_para[str(ind_2)][1])
                    print('=====================')
        # drawing true gaussian functions
        if K_value == 3:
            out_param_true = drawing_Gaussian(true_mu, true_covariance, training_data, true_mu, 2, None)
            # print out parameters of the Gaussian function
            for ind_3 in range(K_value):
                print('Cluster:', ind_3)
                print('Actual Mu:')
                print(out_param_true[str(ind_3)][0])
                print('Actual Covariance:')
                print(out_param_true[str(ind_3)][1])
                print('=====================')
        # drawing log-likelihood values
        drawing_Log_likelihood(best_likelihood)
        
    elif (sys.argv[1] == 'saving'):
        # save the data as a csv file
        save_data_csv_file(training_data)
    else:
        print('Error Input. Please re-choose the task!!')

if __name__ == "__main__":
    main()
