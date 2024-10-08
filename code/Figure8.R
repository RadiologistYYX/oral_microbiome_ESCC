# ##############################################################################
#
##  Figure 8 pheatmap function enrichment
#
# ##############################################################################

# Packages
if (!require("pheatmap")) install.packages("pheatmap")


#  variance explained by study and disease
## import data
plot_df <- read.table("./data/Figure8.txt",sep = "\t)

annotation_r <- data.frame(plot_df[,1])
row.names(annotation_r) <- row.names(plot_df)
ann_colors <- plot_df[,5]
row.names(ann_colors) <- plot_df[,1]

p8 <- pheatmap(plot_df[,c(2,3)],color = colorRampPalette(c("#5A4C88", "white", "#CB422F"))(50), 
              cluster_cols = FALSE, border_color = NA,
              cluster_rows = TRUE, clustering_method = "complete", scale = 'none',
              annotation_row  = annotation_r,   treeheight_row = 120,annotation_colors = ann_colors)

p8

# ##############################################################################
