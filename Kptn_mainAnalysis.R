### Kptn main analysis

require(Seurat)

setwd('/home/jovyan/MH_DDD/')

tab = read.delim('../data/KptnMouse/RNAscope/Objects_Population - Nuclei.txt', sep = '\t', skip = 9)

tab = tab[order(-tab$Position.Y..µm.),]

# Remove nuclei with maximum cross section area greater than 5000:

keep = tab$Nuclei...Nucleus.Volume..µm.. < 4000
tab = tab[keep,]
keep = tab$Nuclei...Nucleus.Volume..µm.. > 1000
tab = tab[keep,]
hist(tab$Nuclei...Nucleus.Volume..µm.., breaks = 100)

hist(log(tab$Nuclei...Intensity.Cell.Atto.490LS.Sum, 2), breaks = 100)
hist(log(tab$Nuclei...Intensity.Nucleus.Atto.490LS.Sum, 2), breaks = 100)
hist(log(tab$Nuclei...Intensity.Cell.Atto.425.Sum, 2), breaks = 100)
hist(log(tab$Nuclei...Intensity.Nucleus.Atto.425.Sum[1:10000], 2), breaks = 50)
hist(log(tab$Nuclei...Intensity.Cell.Alexa.488.Sum, 2), breaks = 100)
hist(log(tab$Nuclei...Intensity.Nucleus.Alexa.488.Sum, 2), breaks = 100)
hist(log(tab$Nuclei...Intensity.Cell.Alexa.568.Sum, 2), breaks = 100)
hist(log(tab$Nuclei...Intensity.Nucleus.Alexa.568.Sum, 2), breaks = 100)
hist(log(tab$Nuclei...Intensity.Cell.Alexa.647.Sum, 2), breaks = 100)
hist(log(tab$Nuclei...Intensity.Nucleus.Alexa.647.Mean[1:10000], 2), breaks = 100)

hist(tab$Nuclei...Cell.Footprint.Area..µm..)

intensityNuclei = tab[,c('Nuclei...Intensity.Nucleus.Atto.490LS.Mean', 'Nuclei...Intensity.Nucleus.Atto.425.Mean',
                         'Nuclei...Intensity.Nucleus.Alexa.488.Mean', 'Nuclei...Intensity.Nucleus.Alexa.568.Mean',
                         'Nuclei...Intensity.Nucleus.Alexa.647.Mean')]

require(ComplexHeatmap)
Heatmap(log(as.matrix(intensityNuclei[1:10000,]),2))

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

### Some more quality control plots:
window = 100
par(mfrow = c(4,1))
plot(tab$Position.Y..µm., tab$Position.X..µm., pch = '.')
plot(tab$Position.Y..µm.[window:length(tab$Position.Y..µm.)], log(rollapply(tab$Nuclei...Intensity.Nucleus.Atto.425.Sum, width = window, by = 1, FUN = mean, align = "left"),2), pch = '.')
plot(tab$Position.Y..µm.[window:length(tab$Position.Y..µm.)], log(rollapply(tab$Nuclei...Intensity.Nucleus.Alexa.488.Sum, width = window, by = 1, FUN = mean, align = "left"),2), pch = '.')
plot(tab$Position.Y..µm.[window:length(tab$Position.Y..µm.)], log(rollapply(as.numeric(tab$Nuclei...Intensity.Nucleus.Alexa.568.Sum), width = window, by = 1, FUN = mean, align = "left"),2), pch = '.')



