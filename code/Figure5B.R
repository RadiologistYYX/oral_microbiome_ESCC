# ##############################################################################
#
##  Figure 5B bar plot
#
# ##############################################################################

# Packages
if (!require("ggthemes")) install.packages("ggthemes")
if (!require("ggplot2")) install.packages("ggplot2")


## import data
markers <- read.table('./data/figure5b.txt',sep ="\t)
markers_plot <- data.frame(family = markers$Family, genus = markers$Genera, species = markers$Species ,fold_change = markers$logfc)
markers_plot$genus <- factor(markers_plot$genus)
markers_plot$family <- factor(markers_plot$family)
markers_plot$species <- factor(markers_plot$species)
# markers_plot1 <- markers_plot
markers_plot1 <- markers_plot
markers_plot$fold_change <- -(markers_plot$fold_change)
markers_plot$fold_change <- scale(markers_plot$fold_change,center = FALSE,scale=max(abs(markers_plot$fold_change)))
markers_plot1 <- markers_plot
markers_plot <- markers_plot %>%
  arrange(desc(fold_change))

markers_plot$ASV1 <- paste0(markers_plot$family," ",markers_plot$genus," ",markers_plot$species)
plot_5b <- ggplot(markers_plot,aes(reorder(ASV1,fold_change), fold_change))+
  geom_bar(aes(fill=factor((fold_change>0)+1)),stat="identity", width=0.7, position=position_dodge(0.7)) +
  coord_flip() +
  scale_fill_manual(values=c("#0072B2", "#D55E00"), guide=FALSE) +
  labs(x="", y="Generalized fold change" ) +
  theme_pander()  +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y  = element_text(hjust=0),
        panel.background = element_rect(fill=NULL, colour = 'white')
  )


plot_5b


# ##############################################################################
