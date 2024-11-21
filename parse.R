library(nflfastR)
library(dplyr, warn.conflicts = FALSE)

### Pull play-level data #######################################################

ids = nflfastR::fast_scraper_schedules(2020:2024) %>%
    dplyr::pull(game_id)
future::plan("multisession")

all_games = nflfastR::build_nflfastR_pbp(ids)
all_games = all_games |> filter(!is.na(posteam), game_id != "2022_20_CIN_BUF")
all_games$away_team[all_games$away_team == "LA"] = "LAR"
all_games$home_team[all_games$home_team == "LA"] = "LAR"
all_games$posteam[all_games$posteam == "LA"] = "LAR"
save(all_games, file="all_games_2024.RData")

### Convert to drive level data ################################################

drives = all_games |> group_by(game_id, drive) |> summarize(
    drive_result = last(fixed_drive_result),
    offense = last(posteam),
    game_type = first(season_type),
    
    away_team = first(away_team),
    home_team = first(home_team),
    
    away_score = max(total_away_score),
    home_score = max(total_home_score),
    
    yards_gained = sum(yards_gained, na.rm=TRUE)
) |> summarize(
    away_team = first(away_team),
    home_team = first(home_team),
    game_type = first(game_type),
    
    away_score = max(away_score),
    home_score = max(home_score),
    away_yards = sum(ifelse(offense==away_team, yards_gained, 0), na.rm=TRUE),
    home_yards = sum(ifelse(offense==home_team, yards_gained, 0), na.rm=TRUE),
    total_yards = sum(yards_gained, na.rm=TRUE),
    
    away_drives = sum(offense == away_team, na.rm=TRUE),
    home_drives = sum(offense == home_team, na.rm=TRUE),
    
    away_touchdowns = sum(offense == away_team & drive_result == 'Touchdown', na.rm=TRUE),
    home_touchdowns = sum(offense == home_team & drive_result == 'Touchdown', na.rm=TRUE),
    away_field_goals = sum(offense == away_team & drive_result == 'Field goal', na.rm=TRUE),
    home_field_goals = sum(offense == home_team & drive_result == 'Field goal', na.rm=TRUE),
    
    away_punts = away_drives - away_touchdowns - away_field_goals,
    home_punts = home_drives - home_touchdowns - home_field_goals
)


### Convert to per-drive data ##################################################

drives$away_td_perc = drives$away_touchdowns / drives$away_drives
drives$away_fg_perc = drives$away_field_goals / drives$away_drives
drives$away_pn_perc = drives$away_punts / drives$away_drives

drives$home_td_perc = drives$home_touchdowns / drives$home_drives
drives$home_fg_perc = drives$home_field_goals / drives$home_drives
drives$home_pn_perc = drives$home_punts / drives$home_drives

### Map last 16 games ##########################################################

drives$season = as.numeric(substr(drives$game_id, 1, 4))
start_2021 = which(drives$season==2021)[1]
drives$away_prev_16 = vector("list", nrow(drives))
drives$home_prev_16 = vector("list", nrow(drives))
for (i in start_2021:nrow(drives)) {
    away_team = drives$away_team[i]
    home_team = drives$home_team[i]
    prev_games = drives[1:(i-1), c("away_team", "home_team")]
    drives$away_prev_16[[i]] = tail(
        which( prev_games$away_team == away_team | prev_games$home_team == away_team ),
        16
    )
    drives$home_prev_16[[i]] = tail(
        which( prev_games$away_team == home_team | prev_games$home_team == home_team ),
        16
    )
}

drives_2020 = drives |> select(-away_prev_16, -home_prev_16) |> filter(season==2020)
drives_2021 = drives |> select(-away_prev_16, -home_prev_16) |> filter(season==2021)
drives_2022 = drives |> select(-away_prev_16, -home_prev_16) |> filter(season==2022)
drives_2023 = drives |> select(-away_prev_16, -home_prev_16) |> filter(season==2023)
drives_2024 = drives |> select(-away_prev_16, -home_prev_16) |> filter(season==2024)

save(drives, file="drives_2024.RData")

### Build time series(es) ######################################################

nr = (nrow(drives) - start_2021 + 1) * 2
touchdown_perc = matrix(0, nrow=nr, ncol=17)
field_goal_perc = matrix(0, nrow=nr, ncol=17)

for (i in start_2021:nrow(drives)) {
    away_prev = drives$away_prev_16[i]
    home_prev = drives$home_prev_16[i]
    row = i - start_2021
    
    for (col in 1:16) {
        away_index = away_prev[[1]][col]
        home_index = home_prev[[1]][col]
        
        touchdown_perc[row*2-1, col] = drives[away_index, "away_td_perc"][[1]]
        touchdown_perc[row*2, col] = drives[home_index, "home_td_perc"][[1]]
        field_goal_perc[row*2-1, col] = drives[away_index, "away_fg_perc"][[1]]
        field_goal_perc[row*2, col] = drives[home_index, "home_fg_perc"][[1]]
    }
    
    touchdown_perc[row*2-1, 17] = drives$away_td_perc[i]
    touchdown_perc[row*2, 17] = drives$home_td_perc[i]
    
    field_goal_perc[row*2-1, 17] = drives$away_fg_perc[i]
    field_goal_perc[row*2, 17] = drives$home_fg_perc[i]
}

write.csv(
    as.data.frame(touchdown_perc),
    "touchdown_perc.csv",
    row.names=FALSE
)
write.csv(
    as.data.frame(field_goal_perc),
    "field_goal_perc.csv",
    row.names=FALSE
)





