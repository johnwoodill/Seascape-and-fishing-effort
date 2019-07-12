library(tidyverse)
library(ggthemes)
library(feather)
library(marmap)
library(ggmap)
library(viridis)
library(lubridate)
library(RColorBrewer)
library(ggpmisc)

GAPI_Key <- file("~/Projects/Anomalous-IUU-Events-Argentina/Google_api_key.txt", "r")
GAPI_Key <- readLines(GAPI_Key)
register_google(key=GAPI_Key)

seascape_labels <- data.frame(seascape = as.factor(seq(1, 33)),
                              nominal = as.factor(c("NORTH ATLANTIC SPRING, ACC TRANSITION",
                                                    "SUBPOLAR TRANSITION",
                                                    "TROPICAL SUBTROPICAL TRANSITION",
                                                    "WESTERN WARM POOL SUBTROPICAL",
                                                    "SUBTROPICAL GYRE TRANSITION",
                                                    "ACC, NUTRIENT STRESS",
                                                    "TEMPERATE TRANSITION",
                                                    "INDOPACIFIC SUBTROPICAL GYRE",
                                                    "EQUATORIAL TRANSITION",
                                                    "HIGHLY OLIGOTROPHIC SUBTROPICAL GYRE",
                                                    "TROPICAL/SUBTROPICAL UPWELLING",
                                                    "SUBPOLAR",
                                                    "SUBTROPICAL GYRE MESOSCALE INFLUENCED",
                                                    "TEMPERATE BLOOMS UPWELLING",
                                                    "TROPICAL SEAS",
                                                    "MEDITTERANEAN RED SEA",
                                                    "SUBTROPICAL TRANSITION \n LOW NUTRIENT STRESS",
                                                    "MEDITTERANEAN RED SEA",
                                                    "ARTIC/ SUBPOLAR SHELVES",
                                                    "SUBTROPICAL, FRESH INFLUENCED COASTAL",
                                                    "WARM, BLOOMS, HIGH NUTS",
                                                    "ARCTIC LATE SUMMER",
                                                    "FRESHWATER INFLUENCED POLAR SHELVES",
                                                    "ANTARCTIC SHELVES",
                                                    "ICE PACK",
                                                    "ANTARCTIC ICE EDGE",
                                                    "HYPERSALINE EUTROPHIC, \n PERSIAN GULF, RED SEA",
                                                    "ARCTIC ICE EDGE","ANTARCTIC",
                                                    "ICE EDGE  BLOOM",
                                                    "1-30% ICE PRESENT",
                                                    "30-80% MARGINAL ICE","PACK ICE")))


file_list <- list.files("~/Projects/Seascape-and-fishing-effort/data/process_CLASS/", full.names = TRUE)

dataset = data.frame()
for (i in file_list){
  indat <- read_feather(i)
  # print(i)
  # print(indat$date[1])
  dataset <- rbind(dataset, indat)
}

sea <- as.data.frame(read_feather("../data/noaa_seascape_Patagonia_Shelf_2012-2016.feather"))

names(sea)[5] <- "seascape"
sea <- filter(sea, !is.na(sea$seascape))
sea$seascape <- factor(sea$seascape, levels = seq(1, 33), labels = c("NORTH ATLANTIC SPRING, ACC TRANSITION",
                                                "SUBPOLAR TRANSITION",
                                                "TROPICAL SUBTROPICAL TRANSITION",
                                                "WESTERN WARM POOL SUBTROPICAL",
                                                "SUBTROPICAL GYRE TRANSITION",
                                                "ACC, NUTRIENT STRESS",
                                                "TEMPERATE TRANSITION",
                                                "INDOPACIFIC SUBTROPICAL GYRE",
                                                "EQUATORIAL TRANSITION",
                                                "HIGHLY OLIGOTROPHIC SUBTROPICAL GYRE",
                                                "TROPICAL/SUBTROPICAL UPWELLING",
                                                "SUBPOLAR",
                                                "SUBTROPICAL GYRE MESOSCALE INFLUENCED",
                                                "TEMPERATE BLOOMS UPWELLING",
                                                "TROPICAL SEAS",
                                                "MEDITTERANEAN RED SEA",
                                                "SUBTROPICAL TRANSITION \n LOW NUTRIENT STRESS",
                                                "MEDITTERANEAN RED SEA",
                                                "ARTIC/ SUBPOLAR SHELVES",
                                                "SUBTROPICAL, FRESH INFLUENCED COASTAL",
                                                "WARM, BLOOMS, HIGH NUTS",
                                                "ARCTIC LATE SUMMER",
                                                "FRESHWATER INFLUENCED POLAR SHELVES",
                                                "ANTARCTIC SHELVES",
                                                "ICE PACK",
                                                "ANTARCTIC ICE EDGE",
                                                "HYPERSALINE EUTROPHIC, \n PERSIAN GULF, RED SEA",
                                                "ARCTIC ICE EDGE","ANTARCTIC",
                                                "ICE EDGE  BLOOM",
                                                "1-30% ICE PRESENT",
                                                "30-80% MARGINAL ICE","PACK ICE"))
sea$date <- as.Date(sea$date)
head(sea)
sea$day <- day(sea$date)
sea$month <- month(sea$date)
sea$year <- year(sea$date)

sea <- left_join(sea, seascape_labels, by = "seascape")

sea <- filter(sea, day == 1 & month == 1 & year == 2016)
head(sea)

dataset$day <- day(dataset$date)
dataset$month <- month(dataset$date)
dataset$year <- year(dataset$date)

gfw <- filter(dataset, day == 1 & month == 1 & year == 2016)
head(gfw)

gfw$seascape <- factor(gfw$seascape, levels = seq(1, 33), labels = c("NORTH ATLANTIC SPRING, ACC TRANSITION",
                                                                     "SUBPOLAR TRANSITION",
                                                                     "TROPICAL SUBTROPICAL TRANSITION",
                                                                     "WESTERN WARM POOL SUBTROPICAL",
                                                                     "SUBTROPICAL GYRE TRANSITION",
                                                                     "ACC, NUTRIENT STRESS",
                                                                     "TEMPERATE TRANSITION",
                                                                     "INDOPACIFIC SUBTROPICAL GYRE",
                                                                     "EQUATORIAL TRANSITION",
                                                                     "HIGHLY OLIGOTROPHIC SUBTROPICAL GYRE",
                                                                     "TROPICAL/SUBTROPICAL UPWELLING",
                                                                     "SUBPOLAR",
                                                                     "SUBTROPICAL GYRE MESOSCALE INFLUENCED",
                                                                     "TEMPERATE BLOOMS UPWELLING",
                                                                     "TROPICAL SEAS",
                                                                     "MEDITTERANEAN RED SEA",
                                                                     "SUBTROPICAL TRANSITION \n LOW NUTRIENT STRESS",
                                                                     "MEDITTERANEAN RED SEA",
                                                                     "ARTIC/ SUBPOLAR SHELVES",
                                                                     "SUBTROPICAL, FRESH INFLUENCED COASTAL",
                                                                     "WARM, BLOOMS, HIGH NUTS",
                                                                     "ARCTIC LATE SUMMER",
                                                                     "FRESHWATER INFLUENCED POLAR SHELVES",
                                                                     "ANTARCTIC SHELVES",
                                                                     "ICE PACK",
                                                                     "ANTARCTIC ICE EDGE",
                                                                     "HYPERSALINE EUTROPHIC, \n PERSIAN GULF, RED SEA",
                                                                     "ARCTIC ICE EDGE","ANTARCTIC",
                                                                     "ICE EDGE  BLOOM",
                                                                     "1-30% ICE PRESENT",
                                                                     "30-80% MARGINAL ICE","PACK ICE"))

dat0 <- dataset %>% 
  filter(!is.na(seascape)) %>% 
  group_by(seascape) %>% 
  summarise(sum_fishing_hours = sum(fishing_hours),
            sum_obs = n()) %>% 
  ungroup() %>% 
  mutate(total_fishing_hours = sum(sum_fishing_hours),
         total_obs = sum(sum_obs)) %>% 
  mutate(p_effort_sea = sum_fishing_hours/total_fishing_hours,
         p_sample_sea = sum_obs/total_obs,
         likelihood = p_effort_sea/p_sample_sea) %>% 
  mutate(seascape = as.factor(seascape)) %>% 
  left_join(seascape_labels, by="seascape") %>% 
  filter(!is.na(seascape)) %>%
  mutate(seascape = reorder(seascape, likelihood))

sum(dat0$p_effort_sea)
sum(dat0$p_sample_sea)

# Annual measure
dat1 <- dataset %>% 
  filter(!is.na(seascape)) %>% 
  group_by(year, seascape) %>% 
  summarise(sum_fishing_hours = sum(fishing_hours),
            sum_obs = sum(n())) %>% 
  ungroup() %>% 
  mutate(total_fishing_hours = sum(sum_fishing_hours),
         total_obs = sum(sum_obs)) %>% 
  mutate(p_effort_sea = sum_fishing_hours/total_fishing_hours,
         p_sample_sea = sum_obs/total_obs,
         likelihood = p_effort_sea/p_sample_sea) %>% 
  mutate(seascape = as.factor(seascape)) %>% 
  left_join(seascape_labels, by="seascape") %>% 
  filter(!is.na(seascape)) %>%
  mutate(seascape = reorder(seascape, likelihood))

# Month measure
dat2 <- dataset %>% 
  filter(!is.na(seascape)) %>% 
  group_by(year, month, seascape) %>% 
  summarise(sum_fishing_hours = sum(fishing_hours),
            sum_obs = sum(n())) %>% 
  ungroup() %>% 
  mutate(total_fishing_hours = sum(sum_fishing_hours),
         total_obs = sum(sum_obs)) %>% 
  mutate(p_effort_sea = sum_fishing_hours/total_fishing_hours,
         p_sample_sea = sum_obs/total_obs,
         likelihood = p_effort_sea/p_sample_sea) %>% 
  mutate(seascape = as.factor(seascape)) %>% 
  left_join(seascape_labels, by="seascape") %>% 
  filter(!is.na(seascape)) %>%
  mutate(seascape = reorder(seascape, likelihood),
         month = factor(month.abb[month], month.abb))


ggplot(dat0, aes(x=seascape, y=likelihood, fill=seascape))  + 
  theme_tufte(12) +
  labs(y="Likelihood Ratio \n (P(effort) / P(sample))", x="Seascape Category", title = "2012-2016") +
  scale_fill_viridis(discrete = TRUE, direction = -1) +
  geom_bar(stat="identity") + 
  coord_flip() + 
  geom_text(aes(label = nominal), hjust=0, size=2.5) +
  theme(legend.position = "none",
        panel.border = element_rect(colour = "grey", fill=NA, size=1)) +
  ylim(0, 1.75) +
  annotate(geom = "table", x = 37, y = -0.8, label = list(dat0), 
           vjust = 1, hjust = 0) +
  # scale_y_continuous(expand = c(0, 0), 
  #                  breaks = c(0, 0.25, 0.5, 0.75, 1),
  #                  labels = c("0", "0.25", "0.5", "0.75", "1"),
  #                  lim=c(0, 1.35)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
NULL

ggsave("~/Projects/Seascape-and-fishing-effort/figures/EDA_aggregate_weighted_fishing_effort.pdf", width=6, height=4)

ggplot(dat1, aes(x=seascape, y=likelihood, fill=seascape))  + 
  theme_tufte(12) +
  labs(y="Likelihood Ratio \n (P(effort) / P(sample))", x="Seascape Category") +
  scale_fill_viridis(discrete = TRUE, direction = -1) +
  geom_bar(stat="identity") + 
  coord_flip() + 
  geom_text(aes(label = nominal), hjust=0, size=2.5) +
  theme(legend.position = "none",
        panel.border = element_rect(colour = "grey", fill=NA, size=1)) +
  # ylim(0, 6.25) +
  scale_y_continuous(expand = c(0, 0),
                   breaks = c(0, 1),
                   labels = c("0", "1"),
                   lim=c(0, 6.5)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  facet_wrap(~year) +
  NULL

ggsave("~/Projects/Seascape-and-fishing-effort/figures/EDA_annual_weighted_fishing_effort.pdf", width=10, height=10)

ggplot(dat2, aes(x=seascape, y=likelihood, fill=seascape))  + 
  theme_tufte(12) +
  labs(y="Likelihood Ratio \n (P(effort) / P(sample))", x="Seascape Category") +
  scale_fill_viridis(discrete = TRUE, direction = -1) +
  geom_bar(stat="identity") + 
  coord_flip() + 
  geom_text(aes(label = nominal), hjust=0, size=2.5) +
  theme(legend.position = "none",
        panel.border = element_rect(colour = "grey", fill=NA, size=1)) +
  # scale_y_continuous(expand = c(0, 0), 
  #                    breaks = c(0, 0.25, 0.5, 0.75, 1),
  #                    labels = c("0", "0.25", "0.5", "0.75", "1"),
  #                    lim=c(0, 2.5)) +
  ylim(0, 5.25) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") + 
  facet_wrap(~month+year, ncol = 5) +
  NULL

ggsave("~/Projects/Seascape-and-fishing-effort/figures/EDA_monthly_weighted_fishing_effort.pdf", width=15, height=40)


# Map
bat <- getNOAA.bathy(-68, -51, -48, -39, res = 1, keep = TRUE)
bat2 <- getNOAA.bathy(-77, -22, -58, -23, res = 1, keep = TRUE)

loc = c(-58, -22)
map1 <- ggmap(get_map(loc, zoom = 3, maptype='toner-background', color='bw', source='stamen')) + 
  geom_segment(x=-62, xend=-62, y=-40, yend=-45, color='red') +
  geom_segment(x=-62, xend=-54, y=-45, yend=-45, color='red') +
  geom_segment(x=-54, xend=-54, y=-45, yend=-40, color='red') +
  geom_segment(x=-62, xend=-54, y=-40, yend=-40, color='red') +
  labs(x=NULL, y=NULL) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL
map1

gfw$seascape = reorder(dat0$seascape, dat0$likelihood)

sea$seascape <- factor(sea$seascape, levels = c("SUBPOLAR TRANSITION",
                                       "TEMPERATE BLOOMS UPWELLING",
                                       "HYPERSALINE EUTROPHIC",
                                       "WARM, BLOOMS, HIGH NUTS",
                                       "TEMPERATE TRANSITION",
                                       "SUBTROPICAL TRANSITION LOW NUTRIENT STRESS",
                                       "SUBPOLAR",
                                       "TROPICAL SEAS",
                                       "SUBTROPICAL GYRE MESOSCALE INFLUENCED",
                                       "TROPICAL SUBTROPICAL TRANSITION",
                                       "INDOPACIFIC SUBTROPICAL GYRE",
                                       "TROPICAL/SUBTROPICAL UPWELLING",
                                       "NORTH ATLANTIC SPRING, ACC TRANSITION"))

map2 <- ggplot(NULL) +
  # scale_shape_discrete(solid=FALSE) +
  labs(title = "January 1, 2016") +
  labs(color='Seascape') +
  theme_tufte(12) +
  geom_point(data=sea, aes(x=degrees_east, y=degrees_north, color=seascape)) +
  # geom_point(data=gfw, aes(x=lon2, y=lat2, color=seascape)) +
  geom_point(data=gfw, aes(x=lon2, y=lat2), shape=1, color="black", alpha = 0.75) +
  labs(x=NULL, y=NULL) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        panel.background = element_rect(fill = 'grey', color = "grey")) +
  # scale_color_viridis(discrete = TRUE, direction = -1) +
  ylim(min(gfw$lat2), max(gfw$lat2)) +
  xlim(min(gfw$lon2), max(gfw$lon2)) +
  scale_color_manual(values = viridis(13), 
                     labels=c("SUBPOLAR TRANSITION",
                              "TEMPERATE BLOOMS UPWELLING",
                              "HYPERSALINE EUTROPHIC",
                              "WARM, BLOOMS, HIGH NUTS",
                              "TEMPERATE TRANSITION",
                              "SUBTROPICAL TRANSITION LOW NUTRIENT STRESS",
                              "SUBPOLAR",
                              "TROPICAL SEAS",
                              "SUBTROPICAL GYRE MESOSCALE INFLUENCED",
                              "TROPICAL SUBTROPICAL TRANSITION",
                              "INDOPACIFIC SUBTROPICAL GYRE",
                              "TROPICAL/SUBTROPICAL UPWELLING",
                              "NORTH ATLANTIC SPRING, ACC TRANSITION")) +
  NULL

map2

ggdraw() + draw_plot(map2, 0, 0, height = 1, width = 1) +
  draw_plot(map1, .326, 0, height = .25, width = .20)

ggsave("~/Projects/Seascape-and-fishing-effort/figures/Patagonia_map.pdf", width=8, height=4)

tbl <- tableGrob(dat0) 
tbl_plot <- grid.arrange(tbl)

ggsave("~/Projects/Seascape-and-fishing-effort/figures/aggregate_table.pdf", plot = tbl_plot, width = 14, height = 5)

library(stargazer)
stargazer(dat0, summarise=FALSE)

system("pdfunite Patagonia_map.pdf EDA_aggregate_weighted_fishing_effort.pdf aggregate_table.pdf EDA_annual_weighted_fishing_effort.pdf EDA_monthly_weighted_fishing_effort.pdf Patagonia_fishing_effort_EDA.pdf")


map3 <- ggplot(NULL) +
  # scale_shape_discrete(solid=FALSE) +
  labs(title = "January 1, 2016") +
  labs(color='Seascape') +
  theme_tufte(12) +
  geom_point(data=filter(sea, seascape == "TEMPERATE BLOOMS UPWELLING"), aes(x=degrees_east, y=degrees_north, color=factor(seascape)))
map3
