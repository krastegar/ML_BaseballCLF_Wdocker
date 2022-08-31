from heapq import merge
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Maxbins that have been renamed to final bins
maxbin_to_dastool = pd.read_csv("/home/bioinfo/Desktop/Kelley_Lab_projects/mse_proj/das_tool_name_key.tsv", sep="\t")
maxbin_to_dastool.rename(columns = {'Original.Bin.Name':'MaxBins', 
                       'Renamed.Bin.Name':'DasTool_Optimized_Bins'}, 
            inplace = True)
#-----using a the "." as a delimeter to remove .fa part from maxbins column
maxbin_to_dastool[["MaxBins", "Junk"]] = maxbin_to_dastool['MaxBins'].str.rsplit('.', n=1, expand=True)
maxbin_to_dastool.drop(columns=maxbin_to_dastool.columns[-1], # removing last column of data frame
        axis=1, 
        inplace=True)
#print(maxbin_to_dastool)

# Creating DataFrame of contigs mapped to the maxbins                                                           #das_tool_output_dir_DASTool_scaffolds2bin.tsv instead
scaffolds_to_maxbins = pd.read_csv("/home/bioinfo/Desktop/Kelley_Lab_projects/mse_proj/das_tool_output_dir_DASTool_scaffolds2bin.txt", sep="\t")
scaffolds_to_maxbins.columns = ["Contig_Nodes", "MaxBins"]
#print(scaffolds_to_maxbins)

# Summary Final bins with Maxbins labels which have # of contigs 
summary_Maxbin_2DasTool = pd.read_csv("/home/bioinfo/Desktop/Kelley_Lab_projects/mse_proj/das_tool_output_dir_DASTool_summary.txt", sep="\t")
summary_Maxbin_2DasTool.rename(columns = {'bin':'MaxBins'}, # renaming bins coloumn so that I can merge them later
            inplace = True)

#print(summary_Maxbin_2DasTool)

#Merged Table of Optimized bins with # of contigs from summary dataframe
merged = pd.merge( summary_Maxbin_2DasTool,maxbin_to_dastool, on="MaxBins")
#print(merged)


#-----------------I think this is good enough for Dr. Kelley (rest is extra credit)----------------

# DataFrame with contigs and the number of reads found in both paired-end metagenomes 
number_of_ContigReads = pd.read_csv("/home/bioinfo/Desktop/Kelley_Lab_projects/mse_proj/quant.sf", sep="\t")
number_of_ContigReads.rename(columns={"Name": "Contig_Nodes"},
                            inplace=True)
#print(number_of_ContigReads)

# have Nodes connected with maxbins and which also contain number of reads
Contig_to_MaxBins_withReads = pd.merge(number_of_ContigReads,scaffolds_to_maxbins, on="Contig_Nodes")
#print(Contig_to_MaxBins_withReads)


# Final DataFrame that has MaxBins, Optimized_Bins, Number of Contigs, Number of Reads (from Salmon)
final = pd.merge(merged, Contig_to_MaxBins_withReads, on="MaxBins")
print(final.head)
print(final.columns)
'''
# Making Boxplots with Seaborn
sns.set(rc = {'figure.figsize':(9,6.5)})
sns.set_theme(style="whitegrid")
#------Looking at number of reads mapped backed to the Metagenome------

#filt = final.loc[(final['NumReads']> 10000)]
final[["DasTool_Optimized_Bins", "Junk"]] = final['DasTool_Optimized_Bins'].str.rsplit('.f', n=1, expand=True)

ax= sns.boxplot(x='DasTool_Optimized_Bins', y = 'NumReads', data=final)
ax.tick_params(labelsize=10)
ax.set_xlabel("Bins", fontsize=15)
ax.set_ylabel("# of Reads", fontsize=15)
#plt.title("Optimized Bins vs Number of Reads", fontdict={'fontsize': 17})
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.savefig("Test.png")
plt.show()

#----Looking at number of Contigs in each min mapped backed to the Metagenome
ax= sns.barplot(x='DasTool_Optimized_Bins', y = 'contigs', data=final)
plt.xticks(fontsize=7.5)
plt.title("Optimized Bins vs Number of Contigs", fontdict={'fontsize': 14})
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()
'''
