#%%  Imports
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, cohen_kappa_score
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.stats import kendalltau, spearmanr, pearsonr
import pingouin as pingouin
from Input.stacked_tiff import load_stacked_tiff
from model import AlexNet

#%% Statistical analyses: Confusion matrix, accuracy, precision and recall 
class Statistical_analysis_confusion_matrix_accuracy_precision_recall:
    """Class of statistical analyses: Confusion matrix, accuracy, precision and recall"""
    
    def __init__(self, score_categories=None, labels_scores=None):
        """Initialise the score-categories and score-labels of the dataset""" 
        
        self.score_categories = score_categories or ['Poor: 0', 'Fair: 1', 'Good: 2', 'Very Good: 3', 'Excellent: 4']
        self.labels_scores = labels_scores or [0, 1, 2, 3, 4]

        self.confusion_matrix = None
        self.results={}         
 
    def create_confusion_matrix(self, rad_scores, mod_scores, normalize=None, figsize=(10,8)):
        """Create a confusion matrix of the model performance on the dataset"""
        
        cm = confusion_matrix(rad_scores, mod_scores, labels=self.labels_scores, normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.score_categories)

        fig, ax = plt.subplots(figsize=figsize)
        disp.plot(ax=ax, cmap='Blues', values_format='d' if normalize is None else '.2f')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Model Quality Score')
        ax.set_ylabel('Radiologists Quality Score')
        plt.tight_layout()

        self.confusion_matrix = cm
        return cm

    def get_confusion_matrix_dataframe(self):
        """Convert confusion matrix to dataframe"""
        
        if self.confusion_matrix is None:        
            raise ValueError("First create Confusion Matrix with -creat_confusion_matrix()- function")

        return pd.DataFrame(
            self.confusion_matrix, 
            index=[f'Rad: {score}' for score in self.score_categories], 
            columns=[f'Mod: {score}' for score in self.score_categories]
        )

    def calculate_accuracy(self, rad_scores, mod_scores):
        """Calculates the total accuracy of the model on the data"""
        
        exact_accuracy = accuracy_score(rad_scores, mod_scores)
        
        accuracy_results ={
            'exact_accuracy': exact_accuracy
        }

        self.results['accuracy'] = accuracy_results
        return accuracy_results
    
    def calculate_precision(self, rad_scores, mod_scores):
        """Calculates macro-, micro-, weighted and per-class precision of the model on the data"""
        
        macro_precision = precision_score(rad_scores, 
                                        mod_scores, 
                                        labels=self.labels_scores,
                                        average='macro',
                                        zero_division=0)
        
        micro_precision = precision_score(rad_scores,
                                        mod_scores,
                                        labels=self.labels_scores,
                                        average='micro',
                                        zero_division=0)

        weighted_precision = precision_score(rad_scores,
                                        mod_scores,
                                        labels=self.labels_scores,
                                        average='weighted',
                                        zero_division=0)

        per_class_precision = precision_score(rad_scores,
                                        mod_scores,
                                        labels=self.labels_scores,
                                        average=None,
                                        zero_division=0)

        class_precision = {}
        for i, (category, precision) in enumerate(zip(self.score_categories, per_class_precision)):
            class_precision[category] = precision

        precision_results = {
            'macro_precision': macro_precision,
            'micro_precision': micro_precision,
            'weighted_precision': weighted_precision,
            'class_precision': class_precision
        }

        self.results['precision'] = precision_results
        return precision_results
    
    def calculate_recall(self, rad_scores, mod_scores):
        """Calculates macro-, micro-, weighted and per-class recall of the model on the data """
        macro_recall = recall_score(rad_scores,
                                    mod_scores,
                                    labels=self.labels_scores,
                                    average='macro',
                                    zero_division=0)

        micro_recall = recall_score(rad_scores,
                                    mod_scores,
                                    labels=self.labels_scores,
                                    average='micro',
                                    zero_division=0)

        weighted_recall = recall_score(rad_scores,
                                       mod_scores,
                                       labels=self.labels_scores,
                                       average='weighted',
                                       zero_division=0)

        per_class_recall = recall_score(rad_scores,
                                        mod_scores,
                                        labels=self.labels_scores,
                                        average=None,
                                        zero_division=0)
        
        class_recall = {}
        for i, (category, recall) in enumerate(zip(self.score_categories, per_class_recall)):
            class_recall[category] = recall
        
        recall_results = {
            'macro_recall': macro_recall,
            'micro_recall': micro_recall,
            'weighted_recall': weighted_recall,
            'class_recall': class_recall
        }

        self.results['recall'] = recall_results
        return recall_results
    
    def evaluate_all(self, rad_scores, mod_scores, show_confusion_matrix=True, figsize=(10,8)):
        """Execute all specified analyses"""
        print("=== Statistical Analyses Results ===\n")

        # Confusion Matrix
        if show_confusion_matrix:
            print("Confusion Matrix:")
            self.create_confusion_matrix(rad_scores, mod_scores, figsize=figsize)
            plt.show()
            print()

        # Accuracy
        print("=== Accuracy ===")
        accuracy_results = self.calculate_accuracy(rad_scores, mod_scores)
        print(f"Accuracy: {accuracy_results['exact_accuracy']:.4f}")
        print()

        # Precision
        print("=== Precision ===")
        precision_results = self.calculate_precision(rad_scores, mod_scores)
        print(f"Macro Precision: {precision_results['macro_precision']:.4f}  (mean precision of all categories)")
        print(f"Micro Precision: {precision_results['micro_precision']:.4f}  (overall precision)")
        print(f"Weighted Precision: {precision_results['weighted_precision']:.4f}  (weighted based on number of samples per score-category)")
        print("\nPrecision per category")
        for category, prec in precision_results['class_precision'].items():
            print(f"    {category}: {prec:.4f}")
        print()

        # Recall 
        print("=== Recall ===")
        recall_results = self.calculate_recall(rad_scores, mod_scores)
        print(f"Macro Recall: {recall_results['macro_recall']:.4f}  (mean recall of all categories)")
        print(f"Micro Recall: {recall_results['micro_recall']:.4f}  (overall recall)")
        print(f"Weighted Recall: {recall_results['weighted_recall']:.4f}  (weighted based on number of samples per score-category)")
        print("\nRecall per category")
        for category, rec in recall_results['class_recall'].items():
            print(f"    {category}: {rec:.4f}")
        print()

        return self.results
    
    def get_summary_dataframe(self):
        """Creates a summarizing Pandas Dataframe of the statistical analyses"""
        
        if not self.results:
            raise ValueError("Eerst -evaluate_all()- functie uitvoeren")
        
        summary_data = []

        for category in self.score_categories:
            row = {
                'Category': category,
                'Accuracy': self.results['accuracy']['exact_accuracy'],
                'Precision': self.results['precision']['class_precision'][category],
                'Recall': self.results['recall']['class_recall'][category]
            }
            summary_data.append(row)

        return pd.DataFrame(summary_data)    


#%% SROCC, KROCC and PLCC
class Statistical_analysis_SROCC_KROCC_PLCC:
    """Class of statistical analysis of SROCC, KROCC and PLCC"""
    
    def __init__(self):
        self.results = {}       

    def calculate_srocc(self, rad_scores, mod_scores):
        """Calculates Spearman's Rank Order Correlation Coefficient"""
        
        srocc, srocc_p_value = spearmanr(mod_scores, rad_scores)

        srocc_results = {
            'coefficient': srocc,
            'p_value': srocc_p_value,
            'description': 'Rang correlation, measures monotone relations, robust for outliers'
        }

        self.results['srocc'] = srocc_results
        return srocc_results
    
    def calculate_krocc(self, rad_scores, mod_scores):
        """Calculates Kendall's Rank Order Correlation Coefficient"""
        
        krocc, krocc_p_value = kendalltau(mod_scores, rad_scores)

        krocc_results = {
            'coefficient': krocc,
            'p_value': krocc_p_value,
            'description': 'Rang correlation, more conservative than SROCC'
        }

        self.results['krocc'] = krocc_results
        return krocc_results

    def calculate_plcc(self, rad_scores, mod_scores):
        """Calculates Pearson Lineair Correlation Coefficient"""
        plcc, plcc_p_value = pearsonr(mod_scores, rad_scores)

        plcc_results = {
            'coefficient': plcc,
            'p_value': plcc_p_value,
            'description': 'Linear correlation, measures how linearly the correlation is'
        }

        self.results['plcc'] = plcc_results
        return plcc_results
    
    def calculate_overall_correlation(self):
        """Overall correlation: SROCC + KROCC + PLCC"""
        
        if not all(key in self.results for key in ['srocc', 'krocc', 'plcc']):
            raise ValueError("First calculate individual coefficients: SROCC, KROCC, PLCC")
        
        overall = (self.results['srocc']['coefficient'] +
                   self.results['krocc']['coefficient'] +
                   self.results['plcc']['coefficient'])
        
        overall_results = {
            'coefficient': overall,
            'description': 'Sum of SROCC + KROCC + PLCC'
        }
        
        self.results['overall'] = overall_results
        return overall_results
    
    
    def analyse_all_correlation_coefficients(self, rad_scores, mod_scores):
        """Analysis of all correlation coefficients"""
        
        print("=== Analysis of correlation coefficients ===")

        # SROCC
        srocc_results = self.calculate_srocc(rad_scores, mod_scores)
        print(f"SROCC: {srocc_results['coefficient']:.4f}, p-value: {srocc_results['p_value']:.4f}")
        print(f"Explanation: {srocc_results['description']}")
        print()

        # KROCC
        krocc_results = self.calculate_krocc(rad_scores, mod_scores)
        print(f"KROCC: {krocc_results['coefficient']:.4f}, p-value: {krocc_results['p_value']:.4f}")
        print(f"Explanation: {krocc_results['description']}")
        print()

        # PLCC
        plcc_results = self.calculate_plcc(rad_scores, mod_scores)
        print(f"PLCC: {plcc_results['coefficient']:.4f}, p-value: {plcc_results['p_value']:.4f}")
        print(f"Explanation: {plcc_results['description']}")
        print()

        # Overall
        overall_results = self.calculate_overall_correlation()
        print(f"Overall Correlation: {overall_results['coefficient']:.4f}")
        print(f"Explanation: {overall_results['description']}")
        print()

        return self.results
    
    def get_correlation_coefficient_dataframe(self):
        """Creates a summarizing Pandas Dataframe of the analysis of correlation coefficients"""

        if not self.results:
            raise ValueError("Eerst -analyse_all_correlation_coefficients- functie uitvoeren")
        
        data = []
        for correlation_type, results in self.results.items():
            if correlation_type == 'overall':
                row = {
                    'Correlation Type': correlation_type.upper(),
                    'Coefficient': results['coefficient'],
                    'P-value': None,    
                    'Description': results['description']
                }
            else:
                row = {
                    'Correlation Type': correlation_type.upper(),
                    'Coefficient': results['coefficient'],
                    'P-value': results['p_value'],    
                    'Description': results['description']
                }
            
            data.append(row)
        
        return pd.DataFrame(data)

#%% Weighted Cohen's Kappa
class weighted_cohens_kappa:
    """Statistical analysis: Cohen's Kappa"""
    
    def __init__(self, score_categories=None, labels_scores=None):
        self.score_categories = score_categories or ['Poor: 0', 'Fair: 1', 'Good: 2', 'Very Good: 3', 'Excellent: 4']
        self.labels_scores = labels_scores or [0, 1, 2, 3, 4]
        self.results = {}

    def calculate_weighted_cohens_kappa(self, rad_scores, mod_scores, weights='quadratic'):
        """Calculates weighted Cohen's Kappa"""
        
        kappa_score = cohen_kappa_score(rad_scores, mod_scores, weights=weights)

        kappa_results = {
            'weighted_kappa': kappa_score,
            'weights_type': weights,
            'interpretation': self.interpret_kappa(kappa_score)                    
        }

        self.results = kappa_results
        return kappa_results
    
    def interpret_kappa(self, kappa_value):
        """Interpretation according to the Landis & Koch guidelines"""

        if kappa_value < 0: 
            return "Worse than coincidental findings ==> None"
        elif kappa_value <= 0.20:
            return "Minimal agreement ==> Slight"
        elif kappa_value <= 0.40:
            return "Reasonable agreement ==> Fair"
        elif kappa_value <= 0.60:
            return "Moderate agreement ==> Moderate"
        elif kappa_value <= 0.80:
            return "Substantial agreement ==> Substantial"
        else:
            return "Almost perfect agreement ==> Almost Perfect"

    def analyse_weighted_cohens_kappa(self, rad_scores, mod_scores, rater1_name="Radiologists", rater2_name="Model"):
        """Analysis of Weighted Cohen's Kappa"""
        
        print("\n=== Weighted Cohen's Kappa Analyse ===")
        print(f"Comparison between {rater1_name} and {rater2_name}")
        print()

        kappa_results = self.calculate_weighted_cohens_kappa(rad_scores, mod_scores, weights='quadratic')

        print(f"Weights Type: {kappa_results['weights_type']}")
        print(f"Weighted Kappa Value: {kappa_results['weighted_kappa']:.4f}")
        print(f"Interpretatie: {kappa_results['interpretation']}")
        print()

        return kappa_results
    
    def create_disagreement_matrix(self, rad_scores, mod_scores):
        """Create a disagreement matrix"""
        
        cm = confusion_matrix(rad_scores, mod_scores, labels=self.labels_scores)

        disagreement_df = pd.DataFrame(
            cm,
            index=[f'Radioloog: {cat}' for cat in self.score_categories],
            columns=[f'Model: {cat}' for cat in self.score_categories]
        )

        return disagreement_df

    def calculate_kappa_confidence_interval(self, rad_scores, mod_scores, weights='quadratic', confidence=0.95):
        """Calculates a Cohen's Kappa Confidence Interval"""
        
        kappa_score = cohen_kappa_score(rad_scores, mod_scores, weights=weights)
        n = len(rad_scores)

        se_approx = np.sqrt(1 / n)  
        alpha = 1 - confidence
        z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645

        margin_error = z_score * se_approx

        ci_results = {
            'kappa': kappa_score,
            'confidence_interval': confidence,
            'lower_bound': kappa_score - margin_error,
            'upper_bound': kappa_score + margin_error,
            'margin_error': margin_error,
            'note': "This is a rough estimation"
        }

        return ci_results
    

#%%
if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Code/AlexNet_v15_fuzzylabels_b128_LR1e-5_WD1e-3.weights.h5")
    
    score_categories  = ['Poor: 0', 'Fair: 1', 'Good: 2', 'Very Good: 3', 'Excellent: 4']
    labels_scores = [0, 1, 2, 3, 4]
    
    val_data_dir_0 = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAC2023_val/LDCTIQAG2023_val/valid_0.tif"
    val_data_dir_1 = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAC2023_val/LDCTIQAG2023_val/valid_1.tif"
    val_labels_file = "C:/Users/lars/OneDrive - Delft University of Technology/Jaargang 3/KTO/Model/Data/LDCTIQAC2023_val/LDCTIQAG2023_val/ground-truth.json"
    
    val_images_0, val_labels_0 = load_stacked_tiff(val_data_dir_0, val_labels_file)
    val_images_1, val_labels_1 = load_stacked_tiff(val_data_dir_1, val_labels_file)

    val_images = np.concatenate([val_images_0, val_images_1], axis=0)
    val_labels = np.concatenate([val_labels_0, val_labels_1], axis=0)

    rad_scores = np.argmax(val_labels, axis=1)
    mod_scores = np.argmax(model.predict(val_images, batch_size=128), axis=1)
    
    # Confusion Matrix, Accuracy, Precision and Recall
    stat_cm_acc_prec_rec = Statistical_analysis_confusion_matrix_accuracy_precision_recall()

    results_cm_acc_prec_rec = stat_cm_acc_prec_rec.evaluate_all(rad_scores, mod_scores)     

    cm_df = stat_cm_acc_prec_rec.get_confusion_matrix_dataframe()      

    summary_cm_acc_prec_rec_df = stat_cm_acc_prec_rec.get_summary_dataframe()      
    #print(summary_cm_acc_prec_rec_df)

    # Correlation Coefficients: SROCC, KROCC, PLCC and Overall 
    correlation_coefficient_analyse = Statistical_analysis_SROCC_KROCC_PLCC()       

    results_correlation_coefficients = correlation_coefficient_analyse.analyse_all_correlation_coefficients(rad_scores, mod_scores)    

    corr_df = correlation_coefficient_analyse.get_correlation_coefficient_dataframe()       
    #print(corr_df)

    # Weighted Cohen's Kappa 
    weighted_cohens_kappa_analyse = weighted_cohens_kappa()

    results_weighted_kappa = weighted_cohens_kappa_analyse.analyse_weighted_cohens_kappa(rad_scores, mod_scores, rater1_name="Radiologists", rater2_name="Model")

    disagreement_matrix = weighted_cohens_kappa_analyse.create_disagreement_matrix(rad_scores, mod_scores)
    # print(disagreement_matrix)

    # Confidence interval
    ci_results = weighted_cohens_kappa_analyse.calculate_kappa_confidence_interval(rad_scores, mod_scores)
    #print(ci_results)
    
