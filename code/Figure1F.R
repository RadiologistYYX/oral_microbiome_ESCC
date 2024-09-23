# ##############################################################################
#
##  Figure 1F Family bar
#
# ##############################################################################

## Packages
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("plyr")) install.packages("plyr")
if (!require("reshape2")) install.packages("reshape2")
if (!require("ggsci")) install.packages("ggsci")


## import data
plot_df <- read.table('Figure1f.txt')
plot_df <- t(plot_df)
plot_bar_df <- plot_df/apply(plot_df, 1, sum)

col.hm <- pal_npg("nrc")(11)

## plot
ggplot(plot_bar_df, aes(x=Var1, y=value, fill=Var2)) +
  geom_bar(stat="identity",position="fill", width=0.8, col='black')+
  theme_pander()+
  scale_fill_manual(values = col.hm) +
  xlab("Group")+ylab("Relative Abundance")+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.x  = element_text(hjust=1, angle=90),
        panel.background = element_rect(fill=NULL, colour = 'white')) +
  guides(fill=guide_legend(title="Family"))+facet_wrap(.~group_info,scales = "free_x")


# ##############################################################################
