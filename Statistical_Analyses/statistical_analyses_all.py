#%% 
# Import packages
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, cohen_kappa_score
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.stats import kendalltau, spearmanr, pearsonr
import pingouin as pingouin
from STACKED_TIFF import load_stacked_tiff
from AlexNet_v15_fuzzylabels_b128_LR1e5_WD1e3 import AlexNet

#%%
# Statische analyse: Confusion matrix, accuracy, precision en recall 

class Statistical_analysis_confusion_matrix_accuracy_precision_recall:
    def __init__(self, score_categories=None, labels_scores=None):
        # Initialiseer de score-categorien en de score-labels
        self.score_categories = score_categories or ['Poor: 0', 'Fair: 1', 'Good: 2', 'Very Good: 3', 'Excellent: 4']
        self.labels_scores = labels_scores or [0, 1, 2, 3, 4]

        self.confusion_matrix = None
        self.results={}         # Dictionary voor de resultaten

    # Confusion matrix van model 
    def create_confusion_matrix(self, rad_scores, mod_scores, normalize=None, figsize=(10,8)):
        cm = confusion_matrix(rad_scores, mod_scores, labels=self.labels_scores, normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.score_categories)

        fig, ax = plt.subplots(figsize=figsize)
        disp.plot(ax=ax, cmap='Blues', values_format='d' if normalize is None else '.2f')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Model Quality Score')
        ax.set_ylabel('Radiologen Quality Score')
        plt.tight_layout()

        self.confusion_matrix = cm
        return cm

    def get_confusion_matrix_dataframe(self):
        if self.confusion_matrix is None:        # Zie bij init, create_confusion_matrix functie nog niet gerund.
            raise ValueError("Eerst Confusion Matrix maken met -creat_confusion_matrix()- functie")

        # Opslaan van de confusion matrix als DataFrame voor indien verdere analyse
        return pd.DataFrame(
            self.confusion_matrix, 
            index=[f'Rad: {score}' for score in self.score_categories], 
            columns=[f'Mod: {score}' for score in self.score_categories]
        )

    
    # Accuracy berekenen
    def calculate_accuracy(self, rad_scores, mod_scores):
        # Exacte Accuracy, dit wordt een percentage exact juiste voorspellingen
        exact_accuracy = accuracy_score(rad_scores, mod_scores)
        
        accuracy_results ={
            'exact_accuracy': exact_accuracy
        }

        self.results['accuracy'] = accuracy_results
        return accuracy_results
    
    
    # Precision berekenen, ook wel de positief voorspellende waarde
    def calculate_precision(self, rad_scores, mod_scores):
        # Macro precision: is de gemiddelde van alle categorien
        macro_precision = precision_score(rad_scores, 
                                        mod_scores, 
                                        labels=self.labels_scores,
                                        average='macro',
                                        zero_division=0)
        
        # Micro precision: de overall precision van het model
        micro_precision = precision_score(rad_scores,
                                        mod_scores,
                                        labels=self.labels_scores,
                                        average='micro',
                                        zero_division=0)

        # Weighted precision: gewogen op basis van aantal samples per klasse
        weighted_precision = precision_score(rad_scores,
                                            mod_scores,
                                            labels=self.labels_scores,
                                            average='weighted',
                                            zero_division=0)


        # Precision per class: de precisie per score-categorie
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
    

    # Recall berekenen, hoe goed model alle positieve klassen weet te vinden, in dit geval wordt score: 5 een TP dan en de rest FP
    def calculate_recall(self, rad_scores, mod_scores):
        # Macro recall: is de gemiddelde van alle categorien 
        macro_recall = recall_score(rad_scores,
                                    mod_scores,
                                    labels=self.labels_scores,
                                    average='macro',
                                    zero_division=0)

        # Micro recall: de overall recall van het model in geheel
        micro_recall = recall_score(rad_scores,
                                    mod_scores,
                                    labels=self.labels_scores,
                                    average='micro',
                                    zero_division=0)

        # Weighted recall: gemiddelde van per-klasse scores, maar gewogen naar support (hoe vaak elke categorien voorkomt)
        weighted_recall = recall_score(rad_scores,
                                       mod_scores,
                                       labels=self.labels_scores,
                                       average='weighted',
                                       zero_division=0)

        # Recall per class: de recall per score-categorie
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
    
    
    # Uitvoeren van confusion matrix, accuracy, precision en recall tegelijkertijd.
    def evaluate_all(self, rad_scores, mod_scores, show_confusion_matrix=True, figsize=(10,8)):
        print("=== Model Statische Analyse Resultaten ===\n")

        # Confusion Matrix
        if show_confusion_matrix:
            print("Confusion Matrix:")
            self.create_confusion_matrix(rad_scores, mod_scores, figsize=figsize)
            plt.show()
            print()

        # Accuracy
        print("=== Accuracy ===")
        accuracy_results = self.calculate_accuracy(rad_scores, mod_scores)
        print(f"Accuracy: {accuracy_results['exact_accuracy']:.3f}")
        print()

        # Precision
        print("=== Precision ===")
        precision_results = self.calculate_precision(rad_scores, mod_scores)
        print(f"Macro Precision: {precision_results['macro_precision']:.3f}  (gemiddelde van alle categoriën)")
        print(f"Micro Precision: {precision_results['micro_precision']:.3f}  (overall precision)")
        print(f"Weighted Precision: {precision_results['weighted_precision']:.3f}  (gewogen op basis van een aantal samples per score-categorie)")
        print("\nPrecision per category")
        for category, prec in precision_results['class_precision'].items():
            print(f"    {category}: {prec:.3f}")
        print()

        # Recall 
        print("=== Recall ===")
        recall_results = self.calculate_recall(rad_scores, mod_scores)
        print(f"Macro Recall: {recall_results['macro_recall']:.3f}  (gemiddelde van alle categoriën)")
        print(f"Micro Recall: {recall_results['micro_recall']:.3f}  (overall recall)")
        print(f"Weighted Recall: {recall_results['weighted_recall']:.3f}  (gewogen op basis van een aantal samples per score-categorie)")
        print("\nRecall per category")
        for category, rec in recall_results['class_recall'].items():
            print(f"    {category}: {rec:.3f}")
        print()

        return self.results
    
    def get_summary_dataframe(self):
        if not self.results:
            raise ValueError("Eerst -evaluate_all()- functie uitvoeren")
        
        # Maak DataFrame met alle metrics per categorie
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


#%%
# Correlatie Coefficiënten: SROCC, KROCC en PLCC

# Belangrijke informatie: Voor de SROCC, KROCC en PLCC is het belangrijk dat je ruwe data als input geeft, hierdoor wordt anders minder nauwkeurig
# en heb je verlies van informatie. De SROCC en KROCC categoriseren zelf in de functie, maar ruwe data als input werkt nog steeds beter. 
# Als er geen ruwe data-continu data is, dan maakt niet uit natuurlijk. 
# Ook kan 1 van de inputs continu (4,3 en 2,8) zijn en de andere input geclassificeerd (4 en 2). 
# Dat maakt niet uit, dan werken de SROCC, KROCC en PLCC functies als nog.  

# p-value: de correlatie tussen de model en radiologen is significant:
# p < 0.001: Zeer sterk bewijs dat je model echt correleert met radioloog beoordelingen
# p < 0.05: Voldoende bewijs voor publicatie/klinisch gebruik
# p ≥ 0.05: Model correlatie is mogelijk niet betrouwbaar 

class Statistical_analysis_SROCC_KROCC_PLCC:
    def __init__(self):
        self.results = {}       # dictionary voor de resultaten

    # SROCC (Spearman's Rank Order Correlation Coefficient):
    # is een rang correlatie, meet monotone relaties, robuust voor uitbijters.
    def calculate_srocc(self, rad_scores, mod_scores):
        srocc, srocc_p_value = spearmanr(mod_scores, rad_scores)

        srocc_results = {
            'coefficient': srocc,
            'p_value': srocc_p_value,
            'description': 'Rang correlatie, meet monotone relaties, robuust voor uitbijters'
        }

        self.results['srocc'] = srocc_results
        return srocc_results
    
    # KROCC (Kendall's Rank Order Correlation Coefficient): 
    # is een rang correlatie, is meer conservatief dan Spearman, beter voor kleine samples. Gevoelig voor wisselingen in rangorde
    def calculate_krocc(self, rad_scores, mod_scores):
        krocc, krocc_p_value = kendalltau(mod_scores, rad_scores)

        krocc_results = {
            'coefficient': krocc,
            'p_value': krocc_p_value,
            'description': 'Rang correlatie, conservatiever dan Spearman, beter voor kleine samples'
        }

        self.results['krocc'] = krocc_results
        return krocc_results

    # PLCC (Pearson Lineair Correlation Coefficient):
    # is een lineaire correlatie, meet hoe goed de data op een rechte lijn ligt. 
    def calculate_plcc(self, rad_scores, mod_scores):
        plcc, plcc_p_value = pearsonr(mod_scores, rad_scores)

        plcc_results = {
            'coefficient': plcc,
            'p_value': plcc_p_value,
            'description': 'Lineaire correlatie, meet hoe goed data op rechte lijn ligt'
        }

        self.results['plcc'] = plcc_results
        return plcc_results
    
    # Overall correlation: SROCC + KROCC + PLCC
    def calculate_overall_correlation(self):
        if not all(key in self.results for key in ['srocc', 'krocc', 'plcc']):
            raise ValueError("Eerst alle individuele correlatie coefficienten berekenen: SROCC, KROCC, PLCC")
        
        overall = (self.results['srocc']['coefficient'] +
                   self.results['krocc']['coefficient'] +
                   self.results['plcc']['coefficient'])
        
        overall_results = {
            'coefficient': overall,
            'description': 'Som van SROCC + KROCC + PLCC'
        }
        
        self.results['overall'] = overall_results
        return overall_results
    
    
    def analyse_all_correlation_coefficients(self, rad_scores, mod_scores):
        print("=== Correlatie Coefficienten Analyse ===")

        # SROCC
        srocc_results = self.calculate_srocc(rad_scores, mod_scores)
        print(f"SROCC: {srocc_results['coefficient']:.4f}, p-value: {srocc_results['p_value']:.4f}")
        print(f"Uitleg: {srocc_results['description']}")
        print()

        # KROCC
        krocc_results = self.calculate_krocc(rad_scores, mod_scores)
        print(f"KROCC: {krocc_results['coefficient']:.4f}, p-value: {krocc_results['p_value']:.4f}")
        print(f"Uitleg: {krocc_results['description']}")
        print()

        # PLCC
        plcc_results = self.calculate_plcc(rad_scores, mod_scores)
        print(f"PLCC: {plcc_results['coefficient']:.4f}, p-value: {plcc_results['p_value']:.4f}")
        print(f"Uitleg: {plcc_results['description']}")
        print()

        # Overall
        overall_results = self.calculate_overall_correlation()
        print(f"Overall Correlation: {overall_results['coefficient']:.4f}")
        print(f"Uitleg: {overall_results['description']}")
        print()

        return self.results
    
    def get_correlation_coefficient_dataframe(self):
        if not self.results:
            raise ValueError("Eerst -analyse_all_correlation_coefficients- functie uitvoeren")
        
        data = []
        for correlation_type, results in self.results.items():
            if correlation_type == 'overall':
                row = {
                    'Correlation Type': correlation_type.upper(),
                    'Coefficient': results['coefficient'],
                    'P-value': None,    # Overall heeft geen p-value
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


#%%
# Correlatie tussen de radiologen in het LUMC. 
# The authors calculated the intraclass correlation coëfficient (ICC) with two-way random effects analysis to assess agreement between the observers, 
# using the Pingouin library. The average radiologist score was used to obtain the ‘ground-truth’ label for each image in both sets

# antwoord wat je eruit krijgt is deze vorm: 
#    Type       ICC         CI95%          F          pval
# 4  ICC2k  0.948183  [0.89, 0.98]  18.754647  3.346482e-12
# Hierbij is de 4 de rij van ICC vormen, dus ICC2k is de 4de rij. De opbouw van rijen is: ICC1, ICC(1,k), enz. 



# Intraclass correlation coëfficient
class intraclass_correlation_coefficient:
    def __init__(self):
        self.dataframe_wide = None
        self.dataframe_long = None
        self.icc_results = None
        self.results = {}

    def create_dataframes(self, image_id, radioloog1, radioloog2):
        # Dataframe in wide format voor in het begin simpel. 
        self.dataframe_wide = pd.DataFrame({
            'Image_id': image_id,
            'Radioloog1': radioloog1,
            'Radioloog2': radioloog2
        })
 
        # Dataframe in long format voor gebruik pingouin. Elke rij is hierbij één beoordeling.
        self.dataframe_long = pd.melt(self.dataframe_wide,
                                      id_vars=['Image_id'],
                                      value_vars=['Radioloog1', 'Radioloog2'],
                                      var_name='Radioloog',
                                      value_name='Score')   # var_name en value_name nodig voor de pingouin.intraclass_corr formule.

        return self.dataframe_wide, self.dataframe_long
    
    def calculate_icc(self, image_id, radioloog1, radioloog2, icc_type='ICC2k'):
        # Maak de dataframes
        self.create_dataframes(image_id, radioloog1, radioloog2)

        # bereken ICC
        self.icc_results = pingouin.intraclass_corr(
            data=self.dataframe_long, 
            targets='Image_id', 
            raters='Radioloog',
            ratings='Score')
        
        # filter op gewenste ICC type ==> 2k
        selected_icc = self.icc_results.query(f"Type == '{icc_type}'")

        if not selected_icc.empty:
            self.results[icc_type] = {
                'icc_value': selected_icc['ICC'].iloc[0],
                'ci_lower': selected_icc['CI95%'].iloc[0][0],
                'ci_upper': selected_icc['CI95%'].iloc[0][1],
                'f_statistic': selected_icc['F'].iloc[0],
                'p_value' : selected_icc['pval'].iloc[0],
                'description': self.get_icc_description(icc_type)                            
            }
        
        return selected_icc
    
    def get_icc_description(self, icc_type):
        descriptions = {
            'ICC1': 'One-way random effects, single measurement',
            'ICC2': 'Two-way random effects, single measurement', 
            'ICC3': 'Two-way mixed effects, single measurement',
            'ICC1k': 'One-way random effects, average of k measurements',
            'ICC2k': 'Two-way random effects, average of k measurements',
            'ICC3k': 'Two-way mixed effects, average of k measurements'
        }
        return descriptions.get(icc_type, 'Onbekend ICC type')

    def analyse_intraclass_correlation_coefficient(self, image_id, radioloog1, radioloog2, icc_type='ICC2k', show_dataframes=False):
        print("=== Intraclass Correlation Coefficient ===")

        icc_result = self.calculate_icc(image_id, radioloog1, radioloog2, icc_type)

        if show_dataframes:
            print("Wide Format DataFrame:")
            print(self.dataframe_wide.head())
            print("\nLong Format DataFrame:")
            print(self.dataframe_long.head())
            print()

        if not icc_result.empty:
            result_data = self.results[icc_type]
            
            print(f"ICC Type: {icc_type}")
            print(f"Beschrijving: {result_data['description']}")
            print(f"ICC Waarde: {result_data['icc_value']:.4f}")
            print(f"95% Betrouwbaarheidsinterval: [{result_data['ci_lower']:.4f}, {result_data['ci_upper']:.4f}]")
            print(f"F-statistiek: {result_data['f_statistic']:.4f}")
            print(f"P-waarde: {result_data['p_value']:.4f}")
            print()
            
            # Toon volledige in een tabel
            print("Volledige ICC Resultaten:")
            print(icc_result[['Type', 'ICC', 'CI95%', 'F', 'pval']])
        else:
            print(f"Geen resultaten gevonden voor ICC type: {icc_type}")
        
        return self.results

    def get_icc_summary_dataframe(self):
        if not self.results:
            raise ValueError("Eerst -analyse_intraclass_correlation_coefficient- functie uitvoeren")
        
        summary_data = []
        for icc_type, result_data in self.results.items():
            row = {
                'ICC_type': icc_type,
                'ICC_value': result_data['icc_value'],
                'CI_lower': result_data['ci_lower'],
                'CI_upper': result_data['ci_upper'],
                'F_statistic': result_data['f_statistic'],
                'P_value': result_data['p_value'],
                'Description': result_data['description']
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)



#%%
# Weighted Cohen's Kappa
# Weighted omdat het een ordinale meervoudige schaal is van 0-4, hiervoor ook quadratische weights, hierdoor grotere penalties bij grotere afwijking. 

class weighted_cohens_kappa:
    def __init__(self, score_categories=None, labels_scores=None):
        self.score_categories = score_categories or ['Poor: 0', 'Fair: 1', 'Good: 2', 'Very Good: 3', 'Excellent: 4']
        self.labels_scores = labels_scores or [0, 1, 2, 3, 4]
        self.results = {}

    def calculate_weighted_cohens_kappa(self, rad_scores, mod_scores, weights='quadratic'):
        # Weights staat voor lineair of quadratic

        kappa_score = cohen_kappa_score(rad_scores, mod_scores, weights=weights)

        kappa_results = {
            'weighted_kappa': kappa_score,
            'weights_type': weights,
            'interpretation': self.interpret_kappa(kappa_score)                    
        }

        self.results = kappa_results
        return kappa_results
    
    def interpret_kappa(self, kappa_value):
        # interpretatie volgens de Landis & Koch richtlijnen 

        if kappa_value < 0: 
            return "Slechter dan toeval ==> None"
        elif kappa_value <= 0.20:
            return "Minimale overeenstemming ==> Slight"
        elif kappa_value <= 0.40:
            return "Redelijke overeenstemming ==> Fair"
        elif kappa_value <= 0.60:
            return "Matige overeenstemming ==> Moderate"
        elif kappa_value <= 0.80:
            return "Aanzienlijke overeenstemming ==> Substantial"
        else:
            return "Bijna perfecte overeenstemming ==> Almost Perfect"

    def analyse_weighted_cohens_kappa(self, rad_scores, mod_scores, rater1_name="Radiologen", rater2_name="Model"):
        print("\n=== Weighted Cohen's Kappa Analyse ===")
        print(f"Vergelijking tussen {rater1_name} en {rater2_name}")
        print()

        # bereken de weighted cohen's kappa quadratic
        kappa_results = self.calculate_weighted_cohens_kappa(rad_scores, mod_scores, weights='quadratic')

        print(f"Weights Type: {kappa_results['weights_type']}")
        print(f"Weighted Kappa Value: {kappa_results['weighted_kappa']:.4f}")
        print(f"Interpretatie: {kappa_results['interpretation']}")
        print()

        return kappa_results
    
    def create_disagreement_matrix(self, rad_scores, mod_scores):
        # Maak confusion matrix
        cm = confusion_matrix(rad_scores, mod_scores, labels=self.labels_scores)

        # Zet over naar een DataFrame
        disagreement_df = pd.DataFrame(
            cm,
            index=[f'Radioloog: {cat}' for cat in self.score_categories],
            columns=[f'Model: {cat}' for cat in self.score_categories]
        )

        return disagreement_df

    def calculate_kappa_confidence_interval(self, rad_scores, mod_scores, weights='quadratic', confidence=0.95):
        kappa_score = cohen_kappa_score(rad_scores, mod_scores, weights=weights)
        n = len(rad_scores)

        # Simplified standard error calculation
        se_approx = np.sqrt(1 / n)  # Zeer grove benadering
        alpha = 1 - confidence
        # Z-score voor confidence interval
        z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645

        margin_error = z_score * se_approx

        ci_results = {
            'kappa': kappa_score,
            'confidence_interval': confidence,
            'lower_bound': kappa_score - margin_error,
            'upper_bound': kappa_score + margin_error,
            'margin_error': margin_error,
            'note': "Dit is een grove benadering"
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
    
    # Confusion Matrix, Accuracy, Precision en Recall
    stat_cm_acc_prec_rec = Statistical_analysis_confusion_matrix_accuracy_precision_recall()    # Initialiseer class

    resultaten_cm_acc_prec_rec = stat_cm_acc_prec_rec.evaluate_all(rad_scores, mod_scores)     # Voer complete evaluatie uit

    cm_df = stat_cm_acc_prec_rec.get_confusion_matrix_dataframe()       # Krijg confusion matrix als DataFrame

    summary_cm_acc_prec_rec_df = stat_cm_acc_prec_rec.get_summary_dataframe()       # Krijg samenvatting DataFrame
    #print(summary_cm_acc_prec_rec_df)

    
    # Correlatie Coefficienten: SROCC, KROCC, PLCC en Overall
    correlation_coefficient_analyse = Statistical_analysis_SROCC_KROCC_PLCC()       # Initialiseer class

    resultaten_correlation_coefficients = correlation_coefficient_analyse.analyse_all_correlation_coefficients(rad_scores, mod_scores)     # Bereken alle correlaties

    corr_df = correlation_coefficient_analyse.get_correlation_coefficient_dataframe()       # Krijg DataFrame
    #print(corr_df)

#%%
    # Weighted Cohen's Kappa quadratisch 
    weighted_cohens_kappa_analyse = weighted_cohens_kappa()

    resultaten_weighted_kappa = weighted_cohens_kappa_analyse.analyse_weighted_cohens_kappa(rad_scores, mod_scores, rater1_name="Radiologen", rater2_name="Model")

    disagreement_matrix = weighted_cohens_kappa_analyse.create_disagreement_matrix(rad_scores, mod_scores)
    # print(disagreement_matrix)

    # Confidence interval
    ci_resultaten = weighted_cohens_kappa_analyse.calculate_kappa_confidence_interval(rad_scores, mod_scores)
    #print(ci_resultaten)


# %%
