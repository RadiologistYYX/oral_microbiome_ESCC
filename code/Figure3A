# ##############################################################################
#
##  Figure 1B variance
#
# ##############################################################################

# Packages
if (!require("pheatmap")) install.packages("pheatmap")
if (!require("patchwork")) install.packages("patchwork")

#  variance explained by study and disease
## import data
heat_df <- read.table("./data/Figure3a.txt")
ann_colors=list(cluster=c(cluster1='grey10',cluster2='grey50',cluster3='grey90'),
                batch = c('PRJNA660092' = '#b4dce7',
                          'PRJNA853196' = '#e6eef1', 'Chongqing Cohort' = '#edc4be'))
ann_row <- readRDS("./data/Figure3a_ann_row.rds")
ann_col <- readRDS("./data/Figure3a_ann_col.rds")

t_heat_df2_1 <- t(heat_df)

bar_plot <- apply(t_heat_df2_1>0,1,sum)/249

bar_plot2 <- apply(t_heat_df2_1>0,2,sum)/43 

up_plot <- barplot(bar_plot2,angle=45,xlab=NULL,ylim=c(0,1),axisnames = FALSE)

right_plot <- barplot(bar_plot,angle=45,xlab=NULL,ylim=c(0,0.3),axisnames = FALSE,border = TRUE)

heat_plot  <- pheatmap(heat_df,cluster_rows = FALSE,cluster_cols = FALSE,show_rownames = FALSE,
                 color = colorRampPalette(colors = c("grey95","black"))(2),scale = 'none',annotation_row = ann_row,
                 legend = FALSE,cutree_cols =3,annotation_col = ann_col,annotation_colors = ann_colors)

heat_plot

# ##############################################################################



