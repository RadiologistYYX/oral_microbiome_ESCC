# ##############################################################################
#
##  Figure 2 variance
#
# ##############################################################################

# Packages
if (!require("pheatmap")) install.packages("pheatmap")
if (!require("patchwork")) install.packages("patchwork")


## import data
df <- read.table("./data/Figure2a.txt",sep="\t")
plot_df <- read.table("./data/Figure2b.txt",header = 1, row.names = 1)



p1 <- ggplot(df,aes(asv,log10FDR,fill=label))+geom_bar(stat = "identity") +
  labs(y = "-log10(FDR)",x="",fill="Biomarkers") + 
  theme(axis.title = element_text(size = 8), axis.text = element_text(angle = 90, size = 8, color ='black'), 
        panel.background = element_blank(),axis.ticks = element_blank(),
        axis.text.x = element_text(hjust = 0),
        panel.border = element_rect(color = "black", size = 2, fill = NA))+scale_x_discrete(position = "top") +
  scale_fill_manual(values = c("#595959", "#C6C6C6"))


p2 <- ggplot(df,aes(asv,gFC_abs,fill=gFC))+
  geom_bar(stat = "identity", position = "stack") +
  labs(y = "generalized Fold Change",x="",fill="Generalized fold change\n(log relative abundance)") +
  scale_fill_gradient2(low="#0A7373",high="#B7BF99",mid="#619986",midpoint = 0)+
  theme(axis.text.x =element_blank(), legend.position="bottom",
        panel.background = element_blank(),axis.ticks = element_blank(),
        axis.title.y = element_text(size = 8),
        panel.border = element_rect(color = "black", size = 2, fill = NA))






abundance <- append(heatmap_df$mean_escc,heatmap_df$mean_normal)
heatmap_df$mean_escc_s <- (heatmap_df$mean_escc-min(abundance))/(max(abundance)-min(abundance))
heatmap_df$mean_normal_s <- (heatmap_df$mean_normal-min(abundance))/(max(abundance)-min(abundance))
heatmap_df <- t(heatmap_df)
heatmap_df <- data.frame(heatmap_df)
annotate_group <- data.frame("mean_escc_s"="escc",
                             "mean_normal_s"="normal")
annotate_group <- t(annotate_group)
annotate_group <- data.frame(annotate_group)
ann_colors=list(p.sig=c("<0.001"='#525252',"0.001~0.01"='#797979',"0.01~0.05"='#a0a0a0',
                        "0.05~0.1"="#c6c6c6",">=0.1"="#ededed"),
                annotate_group = c("escc"="#CC071E","normal"="grey"))

## function prepare to plot heatmap
data_pre <- function(heatmap_data){
  annotate_p <- matrix(ncol = ncol(heatmap_df),nrow=1)
  annotate_p <- data.frame(annotate_p)
  for(i in c(1:ncol(heatmap_df))){
    if(heatmap_df[3,i]<0.001){
      annotate_p[1,i]="<0.001"
    }else if(heatmap_df[3,i]>=0.001 && heatmap_df[3,i]<0.01){
      annotate_p[1,i]="0.001~0.01"
    }else if(heatmap_df[3,i]>=0.01 && heatmap_df[3,i]<0.05){
      annotate_p[1,i]="0.01~0.05"
    }else if(heatmap_df[3,i]>=0.05 && heatmap_df[3,i]<0.1){
      annotate_p[1,i]="0.05~0.1"
    }else{
      annotate_p[1,i]=">=0.1"
    }
  }
  row.names(annotate_p)="p.sig"
  colnames(annotate_p)=colnames(heatmap_data)
  annotate_p <- t(annotate_p)
  annotate_p <- data.frame(annotate_p)
  abundance <- append(heatmap_data$mean_escc,heatmap_data$mean_normal)
  heatmap_data$mean_escc_s <- (heatmap_data$mean_escc-min(abundance))/(max(abundance)-min(abundance))
  heatmap_data$mean_normal_s <- (heatmap_data$mean_normal-min(abundance))/(max(abundance)-min(abundance))
  heatmap_data <- t(heatmap_data)
  heatmap_data <- data.frame(heatmap_data)
  return(heatmap_data)
}

## prepare data for heatmap
plot_df1 <- plot_df[plot_df[,"batch"]=="PRJNA660092"]
plot_df2 <- plot_df[plot_df[,"batch"]=="PRJNA853196"]
plot_df3 <- plot_df[plot_df[,"batch"]=="CQ50"]
plot_df1 <- data_pre(plot_df1)
plot_df2 <- data_pre(plot_df2)
plot_df2 <- data_pre(plot_df2)

p3_1 <- pheatmap(plot_df1[c(4,5),],cluster_rows = F,cluster_cols = F,
         border_color = NA,color = colorRampPalette(colors = c("#0A7373","#B7BF99"))(50),
         scale = "none",cellwidth=8,cellheight =10,show_rownames = F,show_colnames = F,
         annotation_col= annotate_p,annotation_colors = ann_colors,
         annotation_row = annotate_group,annotation_names_row = F,annotation_names_col = F,
         legend = F)

p3_2 <- pheatmap(plot_df2[c(4,5),],cluster_rows = F,cluster_cols = F,
         border_color = NA,color = colorRampPalette(colors = c("#0A7373","#B7BF99"))(50),
         scale = "none",cellwidth=8,cellheight =10,show_rownames = F,show_colnames = F,
         annotation_col= annotate_p,annotation_colors = ann_colors,
         annotation_row = annotate_group,annotation_names_row = F,annotation_names_col = F,
         legend = F)

p3_3 <- pheatmap(plot_df3[c(4,5),],cluster_rows = F,cluster_cols = F,
         border_color = NA,color = colorRampPalette(colors = c("#0A7373","#B7BF99"))(50),
         scale = "none",cellwidth=8,cellheight =10,show_rownames = F,show_colnames = F,
         annotation_col= annotate_p,annotation_colors = ann_colors,
         annotation_row = annotate_group,annotation_names_row = F,annotation_names_col = F,
         legend = F)


p1/p2/p3_1/p3_2/p3_3

# ##############################################################################
