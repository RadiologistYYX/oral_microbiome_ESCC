
# ##############################################################################
#
##  Figure 3B bar plot
#
# ##############################################################################

# Packages
if (!require("ggplot2")) install.packages("ggplot2")


#  variance explained by study and disease
## import data
plot_df <- read.table("./data/Figure3b.txt")

g3b <- ggplot(data=plot_df,aes(x=cluster,y = value,fill=group))+
  geom_boxplot(aes(fill=group))+
  # geom_bar(position = position_dodge(), stat = "identity")+
  # geom_text(aes(label = group, vjust =-0.7, hjust = 0.5, color = obj), show.legend =TRUE)+
  stat_compare_means(aes(group = group),
                     label="p.format",
                     show.legend = F)+theme_classic()+
  scale_fill_manual(values = c("#C2C2C2", "#D9544D"))

ggsave(g3b, filename = './figure/Figure3B.pdf',
       width = 6, height = 6)

# ##########################################################################




