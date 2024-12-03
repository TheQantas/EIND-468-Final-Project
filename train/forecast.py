import sys
import pandas as pd
import numpy as np
from scipy.stats import multinomial

from resources import split_data
from build import evaluate

if __name__ == "__main__":
    [_, model_name, lookup_away, lookup_home] = sys.argv if len(sys.argv) >= 4 else sys.argv + [None, None]

    lookup_away = lookup_away.upper() if lookup_away is not None else lookup_away
    lookup_home = lookup_home.upper() if lookup_home is not None else lookup_home

    full_season = pd.read_csv('../forecast/drives_2024.csv')

    X_train_td_off, _, y_train_td_off, _ = split_data(do_td=True, do_off=True, train_perc=0.8)
    X_train_td_def, _, y_train_td_def, _ = split_data(do_td=True, do_off=False, train_perc=0.8)
    X_train_fg_off, _, y_train_fg_off, _ = split_data(do_td=False, do_off=True, train_perc=0.8)
    X_train_fg_def, _, y_train_fg_def, _ = split_data(do_td=False, do_off=False, train_perc=0.8)

    X_future_td_off = pd.read_csv('../forecast/touchdown_perc_off.csv')
    X_future_td_def = pd.read_csv('../forecast/touchdown_perc_def.csv')
    X_future_fg_off = pd.read_csv('../forecast/field_goal_perc_off.csv')
    X_future_fg_def = pd.read_csv('../forecast/field_goal_perc_def.csv')

    _, _, forecasts_td_off = evaluate(model_name, X_train_td_off, X_future_td_off.iloc[:, 1:].values, y_train_td_off)
    _, _, forecasts_td_def = evaluate(model_name, X_train_td_def, X_future_td_def.iloc[:, 1:].values, y_train_td_def)
    _, _, forecasts_fg_off = evaluate(model_name, X_train_fg_off, X_future_fg_off.iloc[:, 1:].values, y_train_fg_off)
    _, _, forecasts_fg_def = evaluate(model_name, X_train_fg_def, X_future_fg_def.iloc[:, 1:].values, y_train_fg_def)

    full_season['away_td_perc_forecast'] = 0.0
    full_season['away_fg_perc_forecast'] = 0.0
    full_season['home_td_perc_forecast'] = 0.0
    full_season['home_fg_perc_forecast'] = 0.0

    full_season['away_score_forecast'] = 0.0
    full_season['home_score_forecast'] = 0.0

    assert(len(forecasts_td_off) == len(full_season) * 2)

    over_hit = 0
    home_covered = 0
    proj_over_hit = 0
    proj_home_covered = 0
    correct_over_hit = 0
    correct_home_covered = 0

    #                    0%,10%,20%,30%,40%
    hc_games_beyond   = [ 0,  0,  0,  0,  0]
    hc_correct_beyond = [ 0,  0,  0,  0,  0]
    ou_games_beyond   = [ 0,  0,  0,  0,  0]
    ou_correct_beyond = [ 0,  0,  0,  0,  0]

    display_index = None

    for index, row in full_season.iterrows():
        game_id: str = row['game_id']
        away_team = row['away_team']
        away_score = row['away_score']
        home_team = row['home_team']
        home_score = row['home_score']

        if lookup_away is not None and lookup_home is not None and game_id.endswith(lookup_away + '_' + lookup_home):
            display_index = index

        proj_away_td = (forecasts_td_off[index*2][0] + forecasts_td_def[index*2+1][0]) / 2 if model_name != 'lsr' and model_name != 'holt' else (forecasts_td_off[index*2] + forecasts_td_def[index*2+1]) / 2
        proj_home_td = (forecasts_td_off[index*2+1][0] + forecasts_td_def[index*2][0]) / 2 if model_name != 'lsr' and model_name != 'holt' else (forecasts_td_off[index*2+1] + forecasts_td_def[index*2]) / 2
        proj_away_fg = (forecasts_fg_off[index*2][0] + forecasts_fg_def[index*2+1][0]) / 2 if model_name != 'lsr' and model_name != 'holt' else (forecasts_fg_off[index*2] + forecasts_fg_def[index*2+1]) / 2
        proj_home_fg = (forecasts_fg_off[index*2+1][0] + forecasts_fg_def[index*2][0]) / 2 if model_name != 'lsr' and model_name != 'holt' else (forecasts_fg_off[index*2+1] + forecasts_fg_def[index*2]) / 2

        full_season.at[index, 'away_td_perc_forecast'] = proj_away_td
        full_season.at[index, 'home_td_perc_forecast'] = proj_home_td
        full_season.at[index, 'away_fg_perc_forecast'] = proj_away_fg
        full_season.at[index, 'home_fg_perc_forecast'] = proj_home_fg

        proj_away_score = (7 * proj_away_td + 3 * proj_away_fg) * 11
        proj_home_score = (7 * proj_home_td + 3 * proj_home_fg) * 11

        full_season.at[index, 'away_score_forecast'] = proj_away_score
        full_season.at[index, 'home_score_forecast'] = proj_home_score

        iter_home_covered = home_score - away_score > row['home_line']
        iter_over_hit = home_score + away_score > row['over_under']

        if iter_home_covered:
            home_covered += 1
        if iter_over_hit:
            over_hit += 1

        iter_proj_home_covered = proj_home_score - proj_away_score > row['home_line']
        iter_proj_over_hit = proj_home_score + proj_away_score > row['over_under']

        if iter_proj_home_covered:
            proj_home_covered += 1
        if iter_proj_over_hit:
            proj_over_hit += 1

        if iter_over_hit == iter_proj_over_hit:
            correct_over_hit += 1
        if iter_home_covered == iter_proj_home_covered:
            correct_home_covered += 1

        assert(not np.isnan(proj_away_fg))
        assert(not np.isnan(proj_away_td))
        assert(not np.isnan(proj_home_fg))
        assert(not np.isnan(proj_home_td))
        assert(proj_away_fg + proj_away_td >= 0 and proj_away_fg + proj_away_td <= 1)
        assert(proj_home_fg + proj_home_td >= 0 and proj_home_fg + proj_home_td <= 1)

        n = 1000
        ou = 0
        hc = 0

        if model_name == 'holt':
            proj_away_fg = max(proj_away_fg, 0)
            proj_home_fg = max(proj_home_fg, 0)
            proj_away_td = max(proj_away_td, 0)
            proj_home_td = max(proj_home_td, 0)

        try:
            for away_drives, home_drives in zip(
                multinomial.rvs(n=11, p=[proj_away_td, proj_away_fg, 1 - proj_away_td - proj_away_fg], size=n),
                multinomial.rvs(n=11, p=[proj_home_td, proj_home_fg, 1 - proj_home_td - proj_home_fg], size=n)
            ):
                sim_away_score = away_drives[0] * 7 + away_drives[1] * 3
                sim_home_score = home_drives[0] * 7 + home_drives[1] * 3

                if sim_away_score + sim_home_score > row['over_under']:
                    ou += 1
                if sim_home_score - sim_away_score > row['home_line']:
                    hc += 1
        except:
            print([proj_away_td, proj_away_fg, 1 - proj_away_td - proj_away_fg])
            print([proj_home_td, proj_home_fg, 1 - proj_home_td - proj_home_fg])
            raise Exception("Failure")

        ou_diff = abs(ou / n - 0.5) - 0.1
        hc_diff = abs(hc / n - 0.5) - 0.1

        orig_ou_diff = ou_diff
        orig_hc_diff = hc_diff

        ouc = 0
        while ou_diff > 0:
            ou_diff -= 0.1
            ou_games_beyond[ouc] += 1
            if iter_over_hit == iter_proj_over_hit:
                ou_correct_beyond[ouc] += 1
            ouc += 1

        hcc = 0
        while hc_diff > 0:
            hc_diff -= 0.1
            hc_games_beyond[hcc] += 1
            if iter_home_covered == iter_proj_home_covered:
                hc_correct_beyond[hcc] += 1
            hcc += 1
            

    away_td_errors = full_season['away_td_perc_forecast'].to_numpy() - full_season['away_td_perc'].to_numpy()
    home_td_errors = full_season['home_td_perc_forecast'].to_numpy() - full_season['home_td_perc'].to_numpy()
    td_errors = np.concatenate((away_td_errors, home_td_errors))
    
    away_fg_errors = full_season['away_fg_perc_forecast'].to_numpy() - full_season['away_fg_perc'].to_numpy()
    home_fg_errors = full_season['home_fg_perc_forecast'].to_numpy() - full_season['home_fg_perc'].to_numpy()
    fg_errors = np.concatenate((away_fg_errors, home_fg_errors))

    away_score_errors = full_season['away_score_forecast'].to_numpy() - full_season['away_score'].to_numpy()
    home_score_errors = full_season['home_score_forecast'].to_numpy() - full_season['home_score'].to_numpy()
    score_errors = np.concatenate((away_score_errors, home_score_errors))

    print('\n\ntd', 'min', np.min(td_errors), 'max', np.max(td_errors), 'mae', np.mean(np.abs(td_errors)))
    print('fg', 'min', np.min(fg_errors), 'max', np.max(fg_errors), 'mae', np.mean(np.abs(fg_errors)))
    print('score', 'min', np.min(score_errors), 'max', np.max(score_errors), 'mae', np.mean(np.abs(score_errors)))

    total_games = len(full_season)

    print(f'home line actual {(home_covered/total_games):.4} proj {(proj_home_covered/total_games):.4} correct {(correct_home_covered/total_games):.4}')
    print(f'over/under actual {(over_hit/total_games):.4} proj {(proj_over_hit/total_games):.4} correct {(correct_over_hit/total_games):.4}')

    if display_index is not None:
        print(f"{lookup_away}@{lookup_home}:")
        print(full_season.loc[display_index])
    elif lookup_away is not None and lookup_home is not None:
        print(f'Could not find {lookup_away}@{lookup_home}')

    print(hc_correct_beyond, hc_games_beyond)
    print(ou_correct_beyond, ou_games_beyond)







