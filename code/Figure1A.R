




#
setwd("~/batch3_pipline/figure_data/")
library(sf) 
library(cowplot)
China <- sf::st_read("./figure1a_map.json")
Chinacount = China
Chinacount[Chinacount$name=="河南省","number"]=90
Chinacount[Chinacount$name=="江苏省","number"]=109
Chinacount[Chinacount$name=="重庆市","number"]=50
Chinacount$number[is.na(Chinacount$number)]=0
library(ggplot2)
# library(ggspatial)
library(ggthemes)
library(RColorBrewer)
ChinaMap <- Chinacount %>% ggplot() +
  # geom_sf(data = Chinacount, colour = "black",fill="number")+
  geom_sf(aes(geometry = geometry, fill = number), colour = "white")  +
  coord_sf(ylim = c(-2387082,1654989), crs = "+proj=laea +lat_0=40 +lon_0=104")+
  # annotation_scale(location = "bl") +
  scale_fill_gradientn(colours=brewer.pal(11,"RdBu")[4 :8])+
  # annotation_north_arrow(location = "tl", which_north = "false", style = north_arrow_fancy_orienteering)+
  theme(panel.grid = element_blank(),
        panel.background = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank())
SouthChinaSea <- ggplot() +
  geom_sf(data = Chinacount, colour = "#000000", fill="white")+
  coord_sf(xlim = c(117131.4,2115095), ylim = c(-4028017,-1877844),
           crs = "+proj=laea +lat_0=40 +lon_0=104")+
  theme(aspect.ratio = 1.25,
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        panel.grid = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(fill = NA, colour = "#525252"),
        plot.margin = unit(c(0,0,0,0),"mm"))
ggdraw() + 
  draw_plot(ChinaMap) +
  draw_plot(SouthChinaSea, x = 0.88, y = 0.00, width = 0.1, height = 0.3)
pdf('~/batch3_pipline/figure/Figure1a.pdf', useDingbats = FALSE)
ggdraw() + 
  draw_plot(ChinaMap) +
  draw_plot(SouthChinaSea, x = 0.88, y = 0.00, width = 0.1, height = 0.3)
dev.off()
