use baseball; 

DROP TABLE IF EXISTS PitchGameCounts;
CREATE TABLE PitchGameCounts
select
tpc.game_id as pitching_game_id,  
tpc.atBat,
tpc.Hit,
tpc.Hit_By_Pitch,
tpc.Home_Run,
tpc.Strikeout,
tpc.Walk,
tpc.team_id,
tpc.Force_Out,
tpc.Ground_Out,
tpc.Bunt_Ground_out,
tpc.Intent_Walk,
tpc.Fly_Out,
tpc.awayTeam,
tpc.homeTeam, 
tpc.bullpenOutsPlayed,
tpc.bullpenHit,
tpc.bullpenWalk,
tpc.bullpenIntentWalk,
g.game_id as game_id,
g.local_date,
g.home_w,
g.home_l,
g.home_team_id,
b.winner_home_or_away, 
b.game_id as box_game_id,
tbc.awayTeam as TeamBatter_Away,
tbc.homeTeam as TeamBatter_Home, 
tbc.atBat as batter_atBat,
tbc.Hit as batter_Hit,
tbc.Hit_By_Pitch as batter_hitPitch,
tbc.Fly_Out as batter_flyOut,
tbc.Ground_Out as batter_groundOut,
tbc.inning as batter_inning,
tbc.Sac_Fly, 
ts.game_id as streak_game_id,
ts.home_streak,
ts.away_streak,
ts.series_streak
from team_pitching_counts tpc
join game g
on g.game_id = tpc.game_id
join boxscore b 
on b.game_id = g.game_id 
join team_batting_counts tbc 
on tbc.game_id = g.game_id
join team_streak ts 
on ts.game_id = g.game_id ;

-- Adding index's and Engine to speed up query process
-- CREATE UNIQUE INDEX team_game_idx ON PitchGameCounts(team_id, game_id);
-- CREATE UNIQUE INDEX day_of_pitch_idx ON PitchGameCounts(team_id, game_id, local_date);
CREATE INDEX time_idx on PitchGameCounts(local_date);
CREATE INDEX streak_idx on PitchGameCounts(streak_game_id); 
CREATE INDEX game_idx on PitchGameCounts(game_id);
CREATE INDEX pitcher_idx on PitchGameCounts(team_id);
ALTER TABLE PitchGameCounts ENGINE = MyISAM; 

/*
 * this is to see if we have any repeat rows in my original
 * joined tables 
SELECT game_id, count(*)
from PitchGameCounts pgc 
group by game_id 
order by game_id; 

select * from PitchGameCounts pgc 
WHERE game_id = 490;
*/



DROP TABLE IF EXISTS AwayTeam; 
Create Temporary table AwayTeam
SELECT a.team_id,
a.home_team_id,
a.game_id,
a.local_date,
a.winner_home_or_away,
a.awayTeam, 
IFNULL(AVG(a.away_streak/NULLIF(a.series_streak,0)),0) as away_IF,
sum(a.Strikeout) as numberofStrikeouts,
IFNULL(sum(a.bullpenHit)/NULLIF (SUM(a.bullpenOutsPlayed),0),0) as ReliefPitcher_Effic,
IFNULL(sum(a.Strikeout)/NULLIF (SUM(a.Walk),0),0)  as Strike_Walk_Ratio,
IFNULL(SUM(a.Ground_Out)/NULLIF (SUM(a.Fly_out),0),0) as go_fly_ratio,
IFNULL(sum(a.batter_Hit)/NULLIF (sum(a.batter_atBat),0),0) as batting_average,
IFNULL(sum(a.Hit_By_Pitch),0) as Batters_hit_pitch,
IFNULL(sum(a.Home_Run)/SUM(a.Hit),0) as HR_ratio,
IFNULL(sum(a.batter_Hit+a.Walk+a.batter_hitPitch)/NULLIF(sum(a.batter_atBat+a.Walk+a.batter_hitPitch+a.Sac_Fly),0),0) as OBP,
CASE WHEN a.winner_home_or_away = "H" THEN 1 ELSE 0 END as HomeTeamWins
FROM PitchGameCounts as a
JOIN PitchGameCounts as rolling_stat
ON a.team_id  = rolling_stat.team_id 
	AND a.local_date > rolling_stat.local_date 
	AND rolling_stat.local_date BETWEEN a.local_date - INTERVAL 10 DAY 
	AND a.local_date where a.awayTeam = 1 and a.TeamBatter_Away =1
GROUP BY rolling_stat.game_id, rolling_stat.team_id
ORDER BY rolling_stat.local_date DESC; 
 

DROP TABLE IF EXISTS HomeTeam;
Create Temporary table HomeTeam
SELECT a.team_id,
a.home_team_id,
a.game_id,
a.local_date,
a.winner_home_or_away,
a.homeTeam, 
IFNULL(AVG(a.home_streak/NULLIF(a.series_streak,0)),0) as home_IF,
sum(a.Strikeout) as numberofStrikeouts,
IFNULL(sum(a.bullpenHit)/NULLIF (SUM(a.bullpenOutsPlayed),0),0) as ReliefPitcher_Effic,
IFNULL(sum(a.Strikeout)/NULLIF (SUM(a.Walk),0),0)  as Strike_Walk_Ratio,
IFNULL(SUM(a.Ground_Out)/NULLIF (SUM(a.Fly_out),0),0) as go_fly_ratio,
IFNULL(sum(a.batter_Hit)/NULLIF (sum(a.batter_atBat),0),0) as batting_average,
IFNULL(sum(a.Hit_By_Pitch),0) as Batters_hit_pitch,
IFNULL(sum(a.Home_Run)/SUM(a.batter_Hit),0) as HR_ratio,
IFNULL(sum(a.batter_Hit+a.Walk+a.batter_hitPitch)/NULLIF(sum(a.batter_atBat+a.Walk+a.batter_hitPitch+a.Sac_Fly),0),0) as OBP,
CASE WHEN a.winner_home_or_away = "H" THEN 1 ELSE 0 END as homeWins
FROM PitchGameCounts as a
JOIN PitchGameCounts as rolling_stat
ON a.team_id  = rolling_stat.team_id 
	AND a.local_date > rolling_stat.local_date 
	AND rolling_stat.local_date BETWEEN a.local_date - INTERVAL 10 DAY -- Doing stats over 10 days I think gives the model a huge lift
	AND a.local_date where a.homeTeam = 1 and a.TeamBatter_Home =1
GROUP BY rolling_stat.game_id,rolling_stat.team_id
ORDER BY rolling_stat.local_date DESC;


Drop Table if exists FeatureTable; 
CREATE Table FeatureTable
select h.team_id as Home_Team,
a.team_id as Away_Team, 
a.game_id,
-- h.home_IF as Home_IF,
-- a.away_IF as Away_IF,
a.ReliefPitcher_Effic as Away_RelPitch_Ratio,
h.ReliefPitcher_Effic as Home_RelPitch_Ratio,
a.numberofStrikeouts as Away_NumStrikes,
h.numberofStrikeouts as Home_NumStrikes,
a.Strike_Walk_Ratio as Away_StrikeRatio,
h.Strike_Walk_Ratio as Home_StrikeRatio,
a.batting_average as Away_BattingAvg,
h.batting_average as Home_BattingAvg,
a.HR_ratio as Away_HR_Ratio,
h.HR_ratio as Home_HR_Ratio,
a.OBP as Away_OBP,
h.OBP as Home_OBP,
-- ABS(h.home_IF - a.away_IF) as Diff_IF,
ABS(a.ReliefPitcher_Effic -h.ReliefPitcher_Effic) as Diff_ReliefPitch,
ABS(a.numberofStrikeouts-h.numberofStrikeouts) as Diff_NumStrikes,
ABS(a.Strike_Walk_Ratio -h.Strike_Walk_Ratio ) as Diff_Strk_Wlk_Ratio,
ABS(a.batting_average - h.batting_average) as Diff_BA,
ABS(a.HR_ratio-h.HR_ratio) as Diff_HR_ratio,
ABS(a.OBP-h.OBP) as Diff_OBP,
h.homeWins as Home_Team_Wins
from HomeTeam as h
join AwayTeam as a
on a.local_date = h.local_date and a.game_id = h.game_id
GROUP BY a.game_id
ORDER BY h.local_date; 



