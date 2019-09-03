library(tidyverse)
library(ggthemes)
library(stringr)

dat <- read_feather('~/Projects/Seascape-and-fishing-effort/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

dat$illegal <- ifelse(dat$eez == TRUE, ifelse(dat$flag != 'ARG', 1, 0 ), 0)

# Illegal
not_arg = filter(dat, flag != 'ARG' & eez == TRUE)



# Fishing effort EDA
seas <- select(dat, seascape_class)
seas$seascape_class <- ifelse(is.na(seas$seascape_class), 999, seas$seascape_class)
dummies = as.data.frame(model.matrix(~factor(seascape_class), data = seas))
dummies$`(Intercept)` <- NULL
names(dummies) <- gsub('[factor(seascape_class)]', "", x=names(dummies))


# Correlation plot
cdat <- select(dat, fishing_hours, lat1, lon1, length, tonnage, 
               engine_power, sst, sst_grad, sst4, 
               sst4_grad, chlor_a, depth_m, coast_dist_km, illegal)

cdat <- cbind(cdat, dummies)


# cdat %>% 
#   cor(use = 'complete.obs') %>% 
#   corrplot.mixed(upper = "ellipse", tl.cex=.8, tl.pos = 'lt', number.cex = .8)

cdat %>% 
  cor(use = 'complete.obs') %>% 
  ggcorrplot(lab=TRUE, lab_size=3.5, show.legend = FALSE)


# Scatter plot
pdat <- select(dat, fishing_hours, length, tonnage, 
               engine_power, sst, sst_grad, sst4, 
               sst4_grad, chlor_a, depth_m, coast_dist_km)

pdat <- gather(pdat, key = var, value = value, -fishing_hours)

ggplot(pdat, aes(x=value, y=fishing_hours)) + 
         geom_point(size=1, alpha=0.5) +
         geom_smooth(method='lm', color='red') +
         facet_wrap(~var, scales = 'free') +
         NULL













#--------------------------------------------------------------------------

# Illegal EDA
ggplot(dat, aes(x=as.Date(date), y=fishing_hours)) + 
  geom_smooth() +
  theme_tufte(12) +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL


ggplot(dat, aes(x=as.Date(date), y=illegal, fill=factor(month))) + 
  geom_bar(stat='identity') +
  theme_tufte(12) +
  scale_x_date(date_breaks = "6 month", date_labels =  "%b\n%Y") +
  labs(x=NULL, y='# of Illegal Vessels', fill=NULL) +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL


ggplot(filter(dat, illegal == TRUE), aes(x=as.Date(date), y=illegal, fill=factor(month))) + 
  geom_bar(stat='identity') +
  theme_tufte(12) +
  scale_x_date(date_breaks = "6 month", date_labels =  "%b\n%Y") +
  labs(x=NULL, y='# of Illegal Vessels', fill=NULL) +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  facet_wrap(~flag) +
  NULL

ggplot(filter(dat, flag == 'CHN' & illegal == TRUE), aes(x=as.Date(date), y=illegal, fill=factor(month))) + 
  geom_bar(stat='identity') +
  theme_tufte(12) +
  scale_x_date(date_breaks = "6 month", date_labels =  "%b\n%Y") +
  labs(x=NULL, y='# of Illegal Vessels', fill=NULL) +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  facet_wrap(~flag) +
  NULL

# Map of illegal vessels
autoplot.bathy(bat, geom = c("contour", "raster"), coast=TRUE) +
  geom_raster(aes(fill=z)) +
  geom_contour(aes(z = z), color = "white", alpha = 0.01) +
  
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 39, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", "grey50", "grey80")) +
  
  #geom_point(data = dat, aes(x=lon1, y=lat1, color=eez)) +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  geom_point(data = not_arg, aes(x=lon1, y=lat1, color=flag)) +
  
  #geom_point(data = dat, aes(lon1, lat1, fill=fishing_hours)) +
  
  
  # scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 10, 20, 1500)),
  #                      colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", 
  #                                 "grey50", "grey70", "grey85")) +
  # labs(x=NULL, y=NULL, color="km") +
  # geom_segment(data=subdat, aes(x=lon, xend=km_lon, y=lat, yend=km_lat, color=km_cluster), size=0.1) +
  
  # geom_point(data=subdat, aes(x=km_lon, y=km_lat), color='red') +
  # geom_text(data=subdat, aes(x=km_lon, y=km_lat, label = km_avg_dist, vjust=-1), size=3.5) +
  
# geom_text(data=subdat, aes(x=lon, y=lat, label = mmsi, vjust=-1), size=3.5) +
# annotate("text", x=-54.5, y = -39.25, label=date_, size = 4, color='black', fontface=2) +
theme(axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank(),
      legend.direction = 'vertical',
      legend.justification = 'center',
      # legend.position = "none",
      legend.margin=margin(l = 0, unit='cm'),
      legend.text = element_text(size=10),
      legend.title = element_text(size=12),
      panel.grid = element_blank()) +
  # Legend up top
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
  # scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) +
  NULL


