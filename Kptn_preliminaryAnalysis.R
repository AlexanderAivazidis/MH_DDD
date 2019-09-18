### Test Preliminary analysis

require(Seurat)

setwd('/home/jovyan/MH_DDD/')

tab = read.delim('../data/KptnMouse/RNAscope/Objects_Population - Nuclei.txt', sep = '\t', skip = 9)

hist(log(tab$Nuclei...Intensity.Cell.Alexa.488.Sum))
hist(log(tab$Nuclei...Intensity.Cell.Alexa.568.Sum))
hist(log(tab$Nuclei...Intensity.Cell.Alexa.647.Sum))
hist(log(tab$Nuclei...Intensity.Cell.Atto.425.Sum))
hist(log(tab$Nuclei...Intensity.Cell.Atto.490LS.Sum))

tab = t(tab)
colnames(tab) = as.character(1:dim(tab)[2])


intensityFeatures = c('Nuclei...Intensity.Cell.Alexa.488.Sum', 'Nuclei...Intensity.Cell.Alexa.568.Sum', 'Nuclei...Intensity.Cell.Alexa.647.Sum',
                      'Nuclei...Intensity.Cell.Atto.425.Sum', 'Nuclei...Intensity.Cell.Atto.490LS.Sum')
features = rownames(tab)[15:126]
features = intensityFeatures

storage.mode(tab) = 'numeric'

### Subset to make it work faster:

intensityTab = CreateSeuratObject(log(tab[features, 100000:110000]+1))

intensityTab = ScaleData(intensityTab)
intensityTab = RunPCA(intensityTab, features = features)
intensityTab = RunUMAP(intensityTab, dims = 1:4)
UMAPPlot(intensityTab)
PCAPlot(intensityTab)
featurePlots = list()
for (i in 1:length(intensityFeatures)){
  print(i)
  featurePlots[[i]] = FeaturePlot(intensityTab, features = intensityFeatures[[i]])
}
p = cowplot::plot_grid(featurePlots[[1]], featurePlots[[2]], featurePlots[[3]], featurePlots[[4]], 
                   featurePlots[[5]])

pdf(file = 'figures/UMAP_PreliminaryTrial1_allFeatures.pdf', width = 20, height = 10)
print(p)
dev.off()
