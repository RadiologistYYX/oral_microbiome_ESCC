# ##############################################################################
#
##  Figure 1C beta diversity
#
# ##############################################################################

# Packages
if (!require("labdsv")) install.packages("labdsv")
if (!require("coin")) install.packages("coin")
if (!require("vegan")) install.packages("vegan")
if (!require("yaml")) install.packages("yaml")
if (!require("ggpubr")) install.packages("ggpubr")
if (!require("cowplot")) install.packages("cowplot")
if (!require("tidyverse")) install.packages("tidyverse")


#import data
df.plot <- read.table('./data/Figure1c.txt',sep = " ")

#main plot
g.main <- df.plot %>%
  ggplot(aes(x=PCoA1, y=PCoA2, shape= group ,col= batch)) +
  geom_point(size = 1.5 )  +
  labs(x=axis.1.title,
       y=axis.2.title)  +
  scale_colour_manual(values=c('PRJNA660092' = '#010221',
                               'PRJNA853196' = '#0A7373', 'Chongqing Cohort' = '#B7bF99')) +
  scale_shape_manual(values=c(15, 16, 17)) +
  scale_x_continuous(position='top') +
  theme(panel.background = element_rect(fill='white', color = 'black'),
        axis.ticks=element_blank(), axis.text = element_blank(),axis.title.x = element_text(size = 10),
        axis.title.y = element_text(size = 10),
        panel.grid = element_blank(),
        legend.position = c(0.84,0.2),
        legend.text = element_text(size=8),
        legend.title = element_blank())+ annotate("text",x=0.01,y=-0.1,label=paste0("Group:p=",Group_pvalue,"\nBatch:p=",Batch_pvalue))

# study boxplot axis 1
g.s.1 <- df.plot %>% 
  mutate(batch=factor(batch, levels=names(c('PRJNA660092' = '#010221',
                                            'PRJNA853196' = '#0A7373', 'Chongqing Cohort' = '#B7bF99')))) %>% 
  ggplot(aes(y=PCoA1, x=batch, fill=batch)) +
  xlab(paste0('Batch\np=',pvalue.1.batch)) +
  geom_boxplot() +
  scale_fill_manual(values=c('PRJNA660092' = '#010221',
                             'PRJNA853196' = '#0A7373', 'Chongqing Cohort' = '#B7bF99'), guide=FALSE) +
  theme(axis.ticks = element_blank(),
        panel.background = element_rect(fill='white', color = 'black'),
        axis.text = element_blank(), 
        axis.title.x = element_blank(),
        axis.title.y  = element_text(size = 10),
        panel.grid = element_blank()) + 
  coord_flip()


# study boxplot axis 2
g.s.2 <- df.plot %>% 
  mutate(batch=factor(batch, levels=names(c('PRJNA660092' = '#010221',
                                            'PRJNA853196' = '#0A7373', 'Chongqing Cohort' = '#B7bF99')))) %>% 
  ggplot(aes(y=PCoA2, x=batch, fill=batch)) + 
  xlab(paste0('Batch\np=',pvalue.2.batch)) +
  geom_boxplot() + 
  scale_fill_manual(values=c('PRJNA660092' = '#010221',
                             'PRJNA853196' = '#0A7373', 'Chongqing Cohort' = '#B7bF99'), guide = FALSE) +
  
  scale_x_discrete(position='top') +
  theme(axis.ticks=element_blank(), 
        panel.background = element_rect(fill='white', color = 'black'),
        axis.text = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x  = element_text(size = 10),
        panel.grid = element_blank())

# group plot axis1
g.g.1 <- df.plot %>% 
  ggplot(aes(x=group, y=PCoA1, fill=group)) +
  xlab(paste0('Group\np=',pvalue.1.group)) +
  geom_boxplot() +
  scale_fill_manual(values= c('Control' = '#C2C2C2',  'ESCC' = '#d9544d'),guide= FALSE) + 
  ylab(axis.1.group) +
  theme(axis.ticks.y=element_blank(),
        axis.text.x = element_text(size = 10),
        axis.text.y=element_blank(),
        axis.title.y = element_text(size = 10),
        axis.title.x  = element_blank(),
        legend.title=element_text(size =15),legend.text=element_text(size = 15),
        panel.background = element_rect(fill='white', color='black'),
        panel.grid = element_blank()) + 
  coord_flip()

# group plot axis2
g.g.2 <- df.plot %>% 
  ggplot(aes(x=group, y=PCoA2, fill=group)) +
  xlab(paste0('Group\np=',pvalue.2.group)) +
  geom_boxplot() +
  scale_fill_manual(values=c('Control' = '#C2C2C2',  'ESCC' = '#d9544d'), guide=FALSE) + 
  scale_x_discrete(position='top') + 
  scale_y_continuous(position = 'right') +
  ylab(axis.2.group) + 
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank(),
        axis.text.y = element_text(size = 10),
        axis.title.x = element_text(size=10),
        axis.title.y = element_blank(),
        panel.background = element_rect(fill='white', color='black'),
        panel.grid = element_blank())

full <- plot_grid(g.main, g.s.2, g.g.2, g.s.1, NULL,NULL,g.g.1, NULL, NULL,
               nrow=3,
               rel_widths = c(0.8, 0.2, 0.2), rel_heights = c(0.8, 0.2, 0.2))
pdf('figure/Figure1b.pdf', useDingbats = FALSE)
full
dev.off()

# ##############################################################################
