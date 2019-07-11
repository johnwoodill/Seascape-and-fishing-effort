library(tidyverse)
library(ggthemes)
library(feather)

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

setwd("~/Projects/Seascape-and-fishing-effort/data/process_CLASS/")

file_list <- list.files()
for (file in file_list){
  
  # if the merged dataset doesn't exist, create it
  if (!exists("dataset")){
    dataset <- read_feather(file)
  }
  
  # if the merged dataset does exist, append to it
  if (exists("dataset")){
    temp_dataset <-read_feather(file)
    dataset<-rbind(dataset, temp_dataset)
    rm(temp_dataset)
  }
  
}

# Aggregate measure
dat1 <- dataset %>% 
  group_by(seascape) %>% 
  summarise(sum_fishing_hours = sum(fishing_hours),
            sum_obs = sum(n())) %>% 
  ungroup() %>% 
  mutate(total_fishing_hours = sum(sum_fishing_hours),
         total_obs = sum(sum_obs)) %>% 
  mutate(p_effort_sea = sum_fishing_hours/total_fishing_hours,
         p_sample_sea = sum_obs/total_obs,
         weighted_effort = p_effort_sea/p_sample_sea) %>% 
  mutate(seascape = as.factor(seascape)) %>% 
  left_join(seascape_labels, by="seascape") %>% 
  filter(!is.na(seascape)) %>% 
  mutate(seascape = reorder(seascape, weighted_effort))

# Month measure
dat2 <- dataset %>% 
  group_by(month, seascape) %>% 
  summarise(sum_fishing_hours = sum(fishing_hours),
            sum_obs = sum(n())) %>% 
  ungroup() %>% 
  mutate(total_fishing_hours = sum(sum_fishing_hours),
         total_obs = sum(sum_obs)) %>% 
  mutate(p_effort_sea = sum_fishing_hours/total_fishing_hours,
         p_sample_sea = sum_obs/total_obs,
         weighted_effort = p_effort_sea/p_sample_sea) %>% 
  mutate(seascape = as.factor(seascape)) %>% 
  left_join(seascape_labels, by="seascape") %>% 
  filter(!is.na(seascape)) %>% 
  mutate(seascape = reorder(seascape, weighted_effort),
         month = factor(month.abb[month], month.abb))



ggplot(dat1, aes(x=seascape, y=weighted_effort, fill=seascape))  + 
  theme_tufte(12) +
  labs(y="Likelihood Raio \n (P(effort) / P(sample))", x="Seascape Category") +
  scale_fill_viridis(discrete = TRUE, direction = -1) +
  geom_bar(stat="identity") + 
  coord_flip() + 
  geom_text(aes(label = nominal), hjust=0, size=2.5) +
  theme(legend.position = "none",
        panel.border = element_rect(colour = "grey", fill=NA, size=1)) +
  scale_y_continuous(expand = c(0, 0), 
                   breaks = c(0, 0.25, 0.5, 0.75, 1),
                   labels = c("0", "0.25", "0.5", "0.75", "1"),
                   lim=c(0, 1.35)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  # ylim(0, 2) +
  NULL

ggsave("~/Projects/Seascape-and-fishing-effort/figures/EDA_weighted_fishing_effort.pdf", width=8, height=4)

ggplot(dat2, aes(x=seascape, y=weighted_effort, fill=seascape))  + 
  theme_tufte(12) +
  labs(y="Likelihood Raio \n (P(effort) / P(sample))", x="Seascape Category") +
  scale_fill_viridis(discrete = TRUE, direction = -1) +
  geom_bar(stat="identity") + 
  coord_flip() + 
  geom_text(aes(label = nominal), hjust=0, size=2.5) +
  theme(legend.position = "none",
        panel.border = element_rect(colour = "grey", fill=NA, size=1)) +
  scale_y_continuous(expand = c(0, 0), 
                     breaks = c(0, 0.25, 0.5, 0.75, 1),
                     labels = c("0", "0.25", "0.5", "0.75", "1"),
                     lim=c(0, 2.5)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") + facet_wrap(~month) +
  NULL

ggsave("~/Projects/Seascape-and-fishing-effort/figures/EDA_monthly_weighted_fishing_effort.pdf", width=10, height=10)


