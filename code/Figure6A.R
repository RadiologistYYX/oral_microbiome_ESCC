# ##############################################################################
#
##  Figure 6A AUC plot using all ASV features
#
# ##############################################################################

# Packages
if (!require("pheatmap")) install.packages("pheatmap")


## import data
plot_df <- read.table("./data/Figure6a.txt")
p6a <- pheatmap(plot_df,display_numbers = TRUE,cluster_rows = FALSE,cluster_cols = FALSE,color = colorRampPalette(c("#e6eef1", "#edc4be"))(100),
         fontsize = 14, number_color = "black", cellwidth=50, cellheight = 50, border_color = NA,gaps_row = 4,
         main = "The AUC using all ASV features")

p6a

# ##############################################################################
