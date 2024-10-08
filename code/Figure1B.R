# ##############################################################################
#
##  Figure 1B variance
#
# ##############################################################################

# Packages
if (!require("ggplot2")) install.packages("ggplot2")


#  variance explained by study and disease
## import data
df.plot.study <- read.table("./data/Figure1b.txt")

g1b <- df.plot.study %>%
  ggplot(aes(x=disease, y=batch)) +
  geom_point(aes(size=t.mean, fill=meta.sig), shape=21,
             col=alpha(c('black'), alpha=0.4)) +
  xlab(paste0('Variance explained by Disease\n','species',' average: ',
              formatC(mean(df.plot.study$disease)*100, digits=2), '%')) +
  ylab(paste0('Variance explained by Study\n','species',' average: ',
              formatC(mean(df.plot.study$batch)*100, digits=2), '%')) +
  theme_bw() +
  theme(panel.grid.minor = element_blank()
  ) +
  scale_x_continuous(breaks = seq(from=0, to=0.1, by=0.05)) +
  scale_y_continuous(breaks=seq(from=0, to=0.6, by=0.1)) +
  scale_fill_manual(values = alpha(c('grey', '#CC071E'),
                                   alpha=c(0.4, .8)),
                    name=paste0('Significance\n(', alpha.meta,')')) +
  scale_size_area(name='Trimmed mean abundance',
                  breaks=c(1e-05, 1e-03, 1e-02)) +
  guides( size = "legend", colour='legend')

ggsave(g1b, filename = './figure/Figure1B.pdf',
       width = 6, height = 6)

# ##############################################################################
