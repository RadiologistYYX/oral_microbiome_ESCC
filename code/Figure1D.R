# ##############################################################################
#
##  Figure 1D venn plot
#
# ##############################################################################

# Packages
if (!require("VennDiagram")) install.packages("VennDiagram")


## import data
diff_asv <- read.table("./data/Figure1b.txt",sep="\t")
asv_list1 <- diff_asv[,diff_asv[,"batch"=="PRJNA660092"]]
asv_list2 <- diff_asv[,diff_asv[,"batch"=="PRJNA853196"]]
asv_list3 <- diff_asv[,diff_asv[,"batch"=="CQ50"]]

venn.plot <- venn.diagram(
  x = list(
    PRJNA660092 = asv_list1,
    PRJNA853196 = asv_list2,
    Chongqing_Cohort = asv_list3
  ),
  filename = "./figure/Figure1d.tiff",
  col = "transparent",
  fill = c("#010221", "#0A7373", "#B7bF99"),
  alpha = 0.5,
  label.col = c("#F4EEED", "#F4EEED", "#F4EEED", "#F4EEED",
                "#F4EEED", "#F4EEED", "#010221"),
  cex = 1.8,
  fontfamily = "bold",
  fontface = "bold",
  cat.default.pos = "text",
  cat.col = c("#FAF4E3", "#FAF4E3", "#010221"),
  cat.cex = 1.2,
  cat.fontfamily = "bold",
  cat.dist = c(0.06, 0.06, 0.03),
  cat.pos = 0
)



# ##############################################################################
