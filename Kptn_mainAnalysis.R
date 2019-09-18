### Kptn main analysis

require(Seurat)

setwd('/home/jovyan/MH_DDD/')

tab = read.delim('../data/KptnMouse/RNAscope/Objects_Population - Nuclei.txt', sep = '\t', skip = 9)

hist(log(tab$Nuclei...Intensity.Cell.Atto.490LS.Mean, 2))
hist(log(tab$Nuclei...Intensity.Cell.Atto.425.Mean, 2))
hist(log(tab$Nuclei...Intensity.Cell.Alexa.488.Mean, 2))
hist(log(tab$Nuclei...Intensity.Cell.Alexa.568.Mean, 2))
hist(log(tab$Nuclei...Intensity.Cell.Alexa.647.Mean, 2))

# 490LS = Oligodendrocytes
# Alexa568 = Astrocytes
# Alexa488 = GABA
# Alexa647 = Oli progenitors
# Atto425 = Neurons

tab = t(tab)
colnames(tab) = as.character(1:dim(tab)[2])

intensityFeatures = c('Nuclei...Intensity.Cell.Alexa.488.Mean', 'Nuclei...Intensity.Cell.Alexa.568.Mean', 'Nuclei...Intensity.Cell.Alexa.647.Mean',
                      'Nuclei...Intensity.Cell.Atto.425.Mean', 'Nuclei...Intensity.Cell.Atto.490LS.Mean')
celltypes = c('')
subtab = tab[intensityFeatures,]

storage.mode(subtab) = 'numeric'

subtab = log(subtab,10)

maxVector = apply(subtab, 2, function(x) which.max(x))

oligos = t(tab[, maxVector == 5])
plot(oligos[,'Position.X..µm.'], oligos[,'Position.Y..µm.'], pch = '.')

astro = t(tab[, maxVector == 2])
plot(astro[,'Position.X..µm.'], astro[,'Position.Y..µm.'], pch = '.')

### Model with 6 classes: Oligodendrocytes, Oligodendrocyte Precursors, Astrocytes, GABAergic Neurons, Neurons

