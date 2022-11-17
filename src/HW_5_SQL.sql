

use baseball; 

-- show tables; 

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
g.game_id as game_id,
g.local_date,
g.home_w,
g.home_l,
b.winner_home_or_away, 
b.game_id as box_game_id,
tbc.atBat as batter_atBat,
tbc.Hit as batter_Hit,
tbc.Hit_By_Pitch as batter_hitPitch,
tbc.Fly_Out as batter_flyOut,
tbc.Ground_Out as batter_groundOut,
tbc.inning as batter_inning
from team_pitching_counts tpc
join game g
on g.game_id = tpc.game_id
join boxscore b 
on b.game_id = g.game_id 
join team_batting_counts tbc 
on tbc.game_id = g.game_id;

select * from PitchGameCounts pgc order by game_id, team_id ; 
-- Adding index's and Engine to speed up query process
CREATE UNIQUE INDEX team_game_idx ON PitchGameCounts(team_id, game_id);
CREATE UNIQUE INDEX day_of_pitch_idx ON PitchGameCounts(team_id, game_id, local_date);
CREATE INDEX time_idx on PitchGameCounts(local_date);
CREATE INDEX game_idx on PitchGameCounts(game_id);
CREATE INDEX pitcher_idx on PitchGameCounts(team_id);
ALTER TABLE PitchGameCounts ENGINE = MyISAM; 



-- select * from Master; 
DROP TABLE IF EXISTS Master;
CREATE TABLE Master
SELECT a.team_id, 
a.winner_home_or_away,
SUM(a.Walk) as numberOfWalks, -- can add as many features here. 
sum(a.Home_Run) as numberOfHomeRuns,
sum(a.Strikeout) as numberofStrikeouts,
sum(a.Strikeout)/NULLIF (SUM(a.Walk),0)  as Strike_Walk_Ratio,
SUM(a.Ground_Out)/NULLIF (SUM(a.Fly_out),0) as go_fly_ratio,
sum(a.Hit)/NULLIF (sum(atBat),0) as batting average,
sum(a.Hit_By_Pitch) as Batters_hit_pitch,
sum(a.Home_Run)/SUM(a.Hit) as HR_ratio,
sum()
FROM PitchGameCounts as a
JOIN PitchGameCounts as rolling_stat
ON a.team_id  = rolling_stat.team_id 
	AND a.local_date > rolling_stat.local_date 
	AND rolling_stat.local_date BETWEEN a.local_date - INTERVAL 100 DAY 
	AND a.local_date
GROUP BY rolling_stat.team_id 
ORDER BY rolling_stat.local_date DESC;






