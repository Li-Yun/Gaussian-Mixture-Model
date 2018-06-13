import numpy as np

# Gaussian Mixture Model class
class GMM:
    def __init__(self, k_value, repeated_num, epoch_num, in_data):
        self.cluster_num = k_value
        self.repeated_number = repeated_num
        self.maximum_epoch = epoch_num
        self.training_data = in_data
        self.threshold = 1e-8
    # ================ calculate log-likelihood ===============
    def calculate_log_likelihood(self, input_response_matrix):
        return np.sum(np.log(np.sum(input_response_matrix, axis = 1)))                
    # ================ Compute Probability ====================
    def p(self, x, mu, sigma):
        result = np.linalg.det(sigma) ** - 0.5 * (2 * np.pi) ** (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu, np.dot(np.linalg.inv(sigma) , x - mu)))
        return result
    # ================ Training phase =========================
    def model_training(self, in_mean, in_label, in_cov):
        # repeat r times
        all_log_likelihood_collec = []
        parameter_collec = {}
        all_mean_collec = []
        all_cov_collec = []
        all_prior_collec = []
        
        for time_ind in range(self.repeated_number):
            # declare variables
            log_likelihood_value_collec = []
            mean_collec = []
            cov_collec = []
            prior_collec = []
            
            # take the results of Kmean as initial centroid
            gaussian_mean = in_mean
            
            # declare corvariance matrices of all Gaussian functions
            corvariance_matrix = in_cov
            
            # declare initial weights of each Gaussian function
            prior_vector = []
            for priori_ind in range(self.cluster_num):
                number_count = 0
                for data_index in range(self.training_data.shape[0]):
                    if in_label[data_index] == priori_ind:
                        number_count += 1
                prior_vector.append(number_count / self.training_data.shape[0])
            
            # declare response matrix (N by K)
            response_matrix = np.zeros((self.training_data.shape[0], self.cluster_num))

            print('Epoch:', 0)
            log_likelihood_value_collec.append(-10000000000)
            print('Log-likelihood:',-10000000000)
            mean_collec.append(gaussian_mean)
            cov_collec.append(corvariance_matrix)
            prior_collec.append(prior_vector)
            
            # GMM training =============================
            for epoch_ind in range(self.maximum_epoch):
                new_gaussian_mean = np.zeros((self.cluster_num, 2))
                new_corvariance_matrix = []
                new_prior_vector = []
                for cov_ind in range(self.cluster_num):
                    new_corvariance_matrix.append(np.zeros((self.training_data.shape[1], self.training_data.shape[1])))
                    new_prior_vector.append(0.0)           
                print('Epoch:',epoch_ind + 1)
                # E-step =====================================
                # calculate the membership for each cluster
                for data_ind in range(self.training_data.shape[0]):
                    for cluster_ind in range(self.cluster_num):
                        response_matrix[data_ind, cluster_ind] = prior_vector[cluster_ind] * self.p(self.training_data[data_ind, :],
                                                                                                    gaussian_mean[cluster_ind], corvariance_matrix[cluster_ind])
                # compute log-likelihood
                log_likelihood = self.calculate_log_likelihood(response_matrix)
                print('Log-likelihood:',log_likelihood)
                log_likelihood_value_collec.append(log_likelihood)
                
                # normalize response matrix
                temp_array = np.sum(response_matrix, axis = 1)
                for data_ind in range(response_matrix.shape[0]):
                    response_matrix[data_ind, :] = response_matrix[data_ind, :] / temp_array[data_ind]
                
                # compute the number of data points that belong to each cluster
                N_k = np.sum(response_matrix, axis = 0)
                # ============================================
                # M-step =====================================
                # update mean and covariance
                for clus_ind in range(self.cluster_num):
                    # mean
                    new_gaussian_mean[clus_ind] = np.sum(response_matrix[:, clus_ind] * np.transpose(self.training_data), axis = 1) / N_k[clus_ind]
                    
                    # covariance
                    data_mu = self.training_data - gaussian_mean[clus_ind]
                    new_corvariance_matrix[clus_ind] = np.dot(np.transpose(data_mu) * response_matrix[:, clus_ind], data_mu) / N_k[clus_ind]
                
                mean_collec.append(new_gaussian_mean)
                cov_collec.append(new_corvariance_matrix)
                # update the prior
                for clus_index in range(self.cluster_num):
                    new_prior_vector[clus_index] = N_k[clus_index] / self.training_data.shape[0]
                prior_collec.append(new_prior_vector)
                # ============================================
                gaussian_mean = new_gaussian_mean
                corvariance_matrix = new_corvariance_matrix
                prior_vector = new_prior_vector
                # check stopping condition
                if epoch_ind > 0 and np.abs(log_likelihood - log_likelihood_value_collec[epoch_ind - 1]) < self.threshold: break
            # ==========================================
            # store all parameters
            all_log_likelihood_collec.append(log_likelihood_value_collec)
            all_mean_collec.append(mean_collec)
            all_cov_collec.append(cov_collec)
            all_prior_collec.append(prior_collec)
            parameter_collec['mu'] = all_mean_collec
            parameter_collec['cov'] = all_cov_collec
            parameter_collec['prior'] = all_prior_collec
        return all_log_likelihood_collec, parameter_collec
    # ==========================================================
    # fit the true model
    def fit_true_model(self, in_data):
        true_mu = np.zeros((3, 2))
        true_cov = []
        K_1 = in_data[0 : 500, :]
        K_2 = in_data[500 : 1000, :]
        K_3 = in_data[1000 : 1500, :]
        true_mu[1, :] = np.mean(K_1, axis=0)
        true_mu[0, :] = np.mean(K_2, axis=0)
        true_mu[2, :] = np.mean(K_3, axis=0)
        true_cov.append(np.cov(K_2.T))
        true_cov.append(np.cov(K_1.T))
        true_cov.append(np.cov(K_3.T))
        return true_mu, true_cov
    # ================ Testing phase ===========================
    def model_predict(self, trained_mu, trained_cov, trained_weights, data):
        predict_matrix = np.zeros((data.shape[0], len(trained_cov)))
        # calculate the membership for each cluster
        for data_ind in range(data.shape[0]):
            for cluster_ind in range(len(trained_cov)):
                predict_matrix[data_ind, cluster_ind] = trained_weights[cluster_ind] * self.p(data[data_ind, :],
                                                                                              trained_mu[cluster_ind], trained_cov[cluster_ind])
        # normalize response matrix
        temp_array = np.sum(predict_matrix, axis = 1)
        for data_ind in range(predict_matrix.shape[0]):
            predict_matrix[data_ind, :] = predict_matrix[data_ind, :] / temp_array[data_ind]                    
        return predict_matrix
    # =========================================================

