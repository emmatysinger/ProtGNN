from .utils import *
import pickle, os

class TxEval:
    
    def __init__(self, model):
        self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model, self.weight_bias_track, self.wandb = model.df, model.df_train, model.df_valid, model.df_test, model.data_folder, model.G, model.best_model, model.weight_bias_track, model.wandb
        self.device = model.device
        self.disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']
        self.molfunc_rel_types = ['molfunc_protein']
        self.split = model.split

    def load_eval_results(self, save_name):
        if save_name is None:
            save_name = os.path.join(self.data_folder, 'molfunc_centric_eval.pkl')
        else:
            save_name = os.path.join(self.data_folder, save_name)
        with open(save_name, 'rb') as f:
            self.out = pickle.load(f)
        print(self.out)

        
    def eval_disease_centric(self, disease_idxs, relation = None, save_result = False, show_plot = False, verbose = False, save_name = None, return_raw = False, simulate_random = True):
        if self.split == 'full_graph':
            # set only_prediction to True during full graph training
            only_prediction = True
        else:
            only_prediction = False
            
        if disease_idxs == 'test_set':
            disease_idxs = None
        
        self.out = disease_centric_evaluation(self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model,self.device, disease_idxs, relation, self.weight_bias_track, self.wandb, show_plot, verbose, return_raw, simulate_random, only_prediction)
        
        if save_result:
            if save_name is None:
                save_name = os.path.join(self.data_folder, 'disease_centric_eval.pkl')
            with open(save_name, 'wb') as f:
                pickle.dump(self.out, f)
        return self.out
    
    def eval_molfunc_centric(self, molfunc_idxs, relation = None, save_result = False, show_plot = False, verbose = False, save_name = None, return_raw = False, simulate_random = True):
        if self.split == 'full_graph':
            # set only_prediction to True during full graph training
            only_prediction = True
        else:
            only_prediction = False
            
        if molfunc_idxs == 'test_set':
            molfunc_idxs = None

        if show_plot:
            import os
            show_plot = os.path.join(self.data_folder, show_plot)
            if not os.path.exists(show_plot):
                os.mkdir(show_plot)
        
        #self.out = molfunc_centric_evaluation(self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model,self.device, molfunc_idxs, relation, self.weight_bias_track, self.wandb, show_plot, verbose, return_raw, simulate_random, only_prediction)
        self.out = eval_molfunc(self.df, self.df_train, self.df_valid, self.df_test, self.data_folder, self.G, self.best_model,self.device, molfunc_idxs, relation, self.wandb, show_plot, verbose, simulate_random)
        print('Finished eval! Now saving')
        if save_result:
            import pickle, os
            if save_name is None:
                save_name = os.path.join(self.data_folder, 'molfunc_centric_eval.pkl')
            else:
                save_name = os.path.join(self.data_folder, save_name)

            print(save_name)

            with open(save_name, 'wb') as f:
                pickle.dump(self.out, f)
        return self.out
    
    def eval(self):
        print(self.out['rev_molfunc_protein'].columns) #['rev_molfunc_protein']
        ax = self.out['rev_molfunc_protein'].hist(color='steelblue', edgecolor='black',
             grid=False, figsize=(12,8), 
             sharex=False, sharey=False, alpha=0.7)


        # Tight layout to utilize space efficiently
        plt.tight_layout()

        # Save the plot to a PNG file
        print('Saving fig!')
        plt.savefig('/om/user/tysinger/TxGNN/histogram_updated.png')
    
    def retrieve_disease_idxs_test_set(self, relation):
        relation = 'rev_' + relation
        df_train_valid = pd.concat([self.df_train, self.df_valid])
        df_dd = self.df_test[self.df_test.relation.isin(self.disease_rel_types)]
        df_dd_train = df_train_valid[df_train_valid.relation.isin(self.disease_rel_types)]

        df_rel_dd = df_dd[df_dd.relation == relation]        
        return df_rel_dd.x_idx.unique()
    
    def retrieve_molfunc_idxs_test_set(self, relation):
        relation = 'rev_' + relation
        df_train_valid = pd.concat([self.df_train, self.df_valid])
        df_pmf = self.df_test[self.df_test.relation.isin(self.molfunc_rel_types)]
        df_pmf_train = df_train_valid[df_train_valid.relation.isin(self.molfunc_rel_types)]

        df_rel_pmf = df_pmf[df_pmf.relation == relation]        
        return df_rel_pmf.x_idx.unique()