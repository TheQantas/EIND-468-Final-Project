library(nflfastR)
library(dplyr, warn.conflicts = FALSE)

### Pull play-level data #######################################################

if (!exists("all_games")) {
    future::plan("multisession")
    
    ids = nflfastR::fast_scraper_schedules(2020:2024) |>
        dplyr::pull(game_id)
    
    all_games = nflfastR::build_nflfastR_pbp(ids) |> progressr::with_progress()
    
    all_games = all_games |> filter(!is.na(posteam), game_id != "2022_20_CIN_BUF")
    all_games$away_team[all_games$away_team == "LA"] = "LAR"
    all_games$home_team[all_games$home_team == "LA"] = "LAR"
    all_games$posteam[all_games$posteam == "LA"] = "LAR"
    
    save(all_games, file="data/all_games.RData")
    
    rm(ids)
}

### Convert to drive level data ################################################

drives = all_games |> group_by(game_id, drive) |> summarize(
    drive_result = last(fixed_drive_result),
    offense = last(posteam),
    game_type = first(season_type),
    
    away_team = first(away_team),
    home_team = first(home_team),
    
    away_score = max(total_away_score),
    home_score = max(total_home_score),
    
    home_line = mean(spread_line),
    over_under = mean(total_line),
    
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
    home_punts = home_drives - home_touchdowns - home_field_goals,
    
    home_line = mean(home_line),
    over_under = mean(over_under)
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

save(drives, file="data/all_drives.RData")

forecast_drives = drives |> filter(season >= 2023)
drives = drives |> filter(season <= 2023)

save(drives, file="data/drives.RData")
write.csv(as.data.frame(forecast_drives |> select(-away_prev_16, -home_prev_16)), "forecast/drives.csv", row.names=FALSE)

### Build time series(es) for training #########################################

nr = (nrow(drives) - start_2021 + 1) * 2

touchdown_perc_off = matrix(0, nrow=nr, ncol=17)
touchdown_perc_def = matrix(0, nrow=nr, ncol=17)
field_goal_perc_off = matrix(0, nrow=nr, ncol=17)
field_goal_perc_def = matrix(0, nrow=nr, ncol=17)

td_test = vector("numeric", length=nr)
fg_test = vector("numeric", length=nr)

for (i in start_2021:nrow(drives)) {
    away_prev = drives$away_prev_16[i]
    home_prev = drives$home_prev_16[i]
    row = i - start_2021 + 1
    
    for (col in 1:16) {
        away_index = away_prev[[1]][col]
        home_index = home_prev[[1]][col]
        
        away_off_td_column = ifelse(
            drives[away_index, "away_team"] == away_team,
            "away_td_perc",
            "home_td_perc"
        )[[1]]
        away_off_fg_column = ifelse(
            drives[away_index, "away_team"] == away_team,
            "away_fg_perc",
            "home_fg_perc"
        )[[1]]
        home_off_td_column = ifelse(
            drives[home_index, "away_team"] == home_team,
            "away_td_perc",
            "home_td_perc"
        )[[1]]
        home_off_fg_column = ifelse(
            drives[home_index, "away_team"] == home_team,
            "away_fg_perc",
            "home_fg_perc"
        )[[1]]
        
        away_def_td_column = ifelse(
            away_off_td_column == "away_td_perc",
            "home_td_perc",
            "away_td_perc"
        )
        away_def_fg_column = ifelse(
            away_off_fg_column == "away_fg_perc",
            "home_fg_perc",
            "away_fg_perc"
        )
        home_def_td_column = ifelse(
            home_off_td_column == "away_td_perc",
            "home_td_perc",
            "away_td_perc"
        )
        home_def_fg_column = ifelse(
            home_off_fg_column == "away_fg_perc",
            "home_fg_perc",
            "away_fg_perc"
        )
        
        touchdown_perc_off[row*2-1, col] = drives[away_index, away_off_td_column][[1]]
        touchdown_perc_off[row*2, col] = drives[home_index, home_off_td_column][[1]]
        
        touchdown_perc_def[row*2-1, col] = drives[away_index, away_def_td_column][[1]]
        touchdown_perc_def[row*2, col] = drives[home_index, home_def_td_column][[1]]
        
        field_goal_perc_off[row*2-1, col] = drives[away_index, away_off_fg_column][[1]]
        field_goal_perc_off[row*2, col] = drives[home_index, home_off_fg_column][[1]]
        
        field_goal_perc_def[row*2-1, col] = drives[away_index, away_def_fg_column][[1]]
        field_goal_perc_def[row*2, col] = drives[home_index, home_def_fg_column][[1]]
    }
    
    touchdown_perc_off[row*2-1, 17] = drives$away_td_perc[i]
    touchdown_perc_off[row*2, 17] = drives$home_td_perc[i]
    
    touchdown_perc_def[row*2-1, 17] = drives$home_td_perc[i]
    touchdown_perc_def[row*2, 17] = drives$away_td_perc[i]
    
    field_goal_perc_off[row*2-1, 17] = drives$away_fg_perc[i]
    field_goal_perc_off[row*2, 17] = drives$home_fg_perc[i]
    
    field_goal_perc_def[row*2-1, 17] = drives$home_fg_perc[i]
    field_goal_perc_def[row*2, 17] = drives$away_fg_perc[i]
    
    away_td_exp = (mean(touchdown_perc_off[row*2-1, 1:16]) + mean(touchdown_perc_def[row*2, 1:16])) / 2
    home_td_exp = (mean(touchdown_perc_off[row*2, 1:16]) + mean(touchdown_perc_def[row*2-1, 1:16])) / 2

    td_test[i] = abs(drives$away_td_perc[i] - away_td_exp)
    td_test[i+1] = abs(drives$home_td_perc[i] - home_td_exp)
    
    
}

write.csv(
    as.data.frame(touchdown_perc_off),
    "series/touchdown_perc_off.csv",
    row.names=FALSE
)
write.csv(
    as.data.frame(field_goal_perc_off),
    "series/field_goal_perc_off.csv",
    row.names=FALSE
)
write.csv(
    as.data.frame(touchdown_perc_def),
    "series/touchdown_perc_def.csv",
    row.names=FALSE
)
write.csv(
    as.data.frame(field_goal_perc_def),
    "series/field_goal_perc_def.csv",
    row.names=FALSE
)

### Build time series(es) for forecasting ######################################

start_2024 = which(forecast_drives$season == 2024)[1]
nr2 = (nrow(forecast_drives) - start_2024 + 1) * 2
offset = nrow(drives |> filter(season < 2023))

touchdown_perc_off2 = matrix(0, nrow=nr2, ncol=17)
touchdown_perc_def2 = matrix(0, nrow=nr2, ncol=17)
field_goal_perc_off2 = matrix(0, nrow=nr2, ncol=17)
field_goal_perc_def2 = matrix(0, nrow=nr2, ncol=17)

for (i in start_2024:nrow(forecast_drives)) {
    away_team = forecast_drives$away_team[i]
    home_team = forecast_drives$home_team[i]
    game_id = forecast_drives$game_id[i]
    
    away_prev = forecast_drives$away_prev_16[i]
    home_prev = forecast_drives$home_prev_16[i]
    
    row = i - start_2024 + 1
    
    touchdown_perc_off2[row*2-1, 1] = paste0(game_id, "-", away_team)
    touchdown_perc_off2[row*2, 1] = paste0(game_id, "-", home_team)
    touchdown_perc_def2[row*2-1, 1] = paste0(game_id, "-", away_team)
    touchdown_perc_def2[row*2, 1] = paste0(game_id, "-", home_team)
    
    field_goal_perc_off2[row*2-1, 1] = paste0(game_id, "-", away_team)
    field_goal_perc_off2[row*2, 1] = paste0(game_id, "-", home_team)
    field_goal_perc_def2[row*2-1, 1] = paste0(game_id, "-", away_team)
    field_goal_perc_def2[row*2, 1] = paste0(game_id, "-", home_team)
    
    for (col in 2:17) {
        away_index = away_prev[[1]][col-1] - offset + 1
        home_index = home_prev[[1]][col-1] - offset + 1
        
        away_off_td_column = ifelse(
            drives[away_index, "away_team"] == away_team,
            "away_td_perc",
            "home_td_perc"
        )[[1]]
        away_off_fg_column = ifelse(
            drives[away_index, "away_team"] == away_team,
            "away_fg_perc",
            "home_fg_perc"
        )[[1]]
        home_off_td_column = ifelse(
            drives[home_index, "away_team"] == home_team,
            "away_td_perc",
            "home_td_perc"
        )[[1]]
        home_off_fg_column = ifelse(
            drives[home_index, "away_team"] == home_team,
            "away_fg_perc",
            "home_fg_perc"
        )[[1]]
        
        away_def_td_column = ifelse(
            away_off_td_column == "away_td_perc",
            "home_td_perc",
            "away_td_perc"
        )
        away_def_fg_column = ifelse(
            away_off_fg_column == "away_fg_perc",
            "home_fg_perc",
            "away_fg_perc"
        )
        home_def_td_column = ifelse(
            home_off_td_column == "away_td_perc",
            "home_td_perc",
            "away_td_perc"
        )
        home_def_fg_column = ifelse(
            home_off_fg_column == "away_fg_perc",
            "home_fg_perc",
            "away_fg_perc"
        )
        
        if (is.na(forecast_drives[away_index, away_off_td_column][[1]])) {
            print(away_index)
            print(away_off_td_column)
            stop()
        }
        
        touchdown_perc_off2[row*2-1, col] = forecast_drives[away_index, away_off_td_column][[1]]
        touchdown_perc_off2[row*2, col] = forecast_drives[home_index, home_off_td_column][[1]]
        
        touchdown_perc_def2[row*2-1, col] = forecast_drives[away_index, away_def_td_column][[1]]
        touchdown_perc_def2[row*2, col] = forecast_drives[home_index, home_def_td_column][[1]]
        
        field_goal_perc_off2[row*2-1, col] = forecast_drives[away_index, away_off_fg_column][[1]]
        field_goal_perc_off2[row*2, col] = forecast_drives[home_index, home_off_fg_column][[1]]
        
        field_goal_perc_def2[row*2-1, col] = forecast_drives[away_index, away_def_fg_column][[1]]
        field_goal_perc_def2[row*2, col] = forecast_drives[home_index, home_def_fg_column][[1]]
    }
    
}

stopifnot(!any(is.na(touchdown_perc_off2)))

write.csv(
    as.data.frame(touchdown_perc_off2),
    "forecast/touchdown_perc_off.csv",
    row.names=FALSE
)
write.csv(
    as.data.frame(field_goal_perc_off2),
    "forecast/field_goal_perc_off.csv",
    row.names=FALSE
)
write.csv(
    as.data.frame(touchdown_perc_def2),
    "forecast/touchdown_perc_def.csv",
    row.names=FALSE
)
write.csv(
    as.data.frame(field_goal_perc_def2),
    "forecast/field_goal_perc_def.csv",
    row.names=FALSE
)




