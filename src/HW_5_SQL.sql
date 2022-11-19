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
g.home_team_id,
b.winner_home_or_away, 
b.game_id as box_game_id,
tbc.atBat as batter_atBat,
tbc.Hit as batter_Hit,
tbc.Hit_By_Pitch as batter_hitPitch,
tbc.Fly_Out as batter_flyOut,
tbc.Ground_Out as batter_groundOut,
tbc.inning as batter_inning,
tbc.Sac_Fly 
from team_pitching_counts tpc
join game g
on g.game_id = tpc.game_id
join boxscore b 
on b.game_id = g.game_id 
join team_batting_counts tbc 
on tbc.game_id = g.game_id;

select * from PitchGameCounts pgc order by game_id, team_id ; 
-- Adding index's and Engine to speed up query process
-- CREATE UNIQUE INDEX team_game_idx ON PitchGameCounts(team_id, game_id);
-- CREATE UNIQUE INDEX day_of_pitch_idx ON PitchGameCounts(team_id, game_id, local_date);
CREATE INDEX time_idx on PitchGameCounts(local_date);
CREATE INDEX game_idx on PitchGameCounts(game_id);
CREATE INDEX pitcher_idx on PitchGameCounts(team_id);
ALTER TABLE PitchGameCounts ENGINE = MyISAM; 



-- select * from Master; 
DROP TABLE IF EXISTS Master;
CREATE TABLE Master
SELECT a.team_id,
a.home_team_id,
a.game_id,
a.local_date,
a.winner_home_or_away,
SUM(a.Walk) as numberOfWalks, -- can add as many features here. 
sum(a.Home_Run) as numberOfHomeRuns,
sum(a.Strikeout) as numberofStrikeouts,
IFNULL(sum(a.Strikeout)/NULLIF (SUM(a.Walk),0),0)  as Strike_Walk_Ratio,
IFNULL(SUM(a.Ground_Out)/NULLIF (SUM(a.Fly_out),0),0) as go_fly_ratio,
IFNULL(sum(a.Hit)/NULLIF (sum(a.atBat),0),0) as batting_average,
IFNULL(sum(a.Hit_By_Pitch),0) as Batters_hit_pitch,
IFNULL(sum(a.Home_Run)/SUM(a.Hit),0) as HR_ratio,
IFNULL(sum(a.Hit+a.Walk+a.Hit_By_Pitch)/NULLIF(sum(a.atBat+a.Walk+a.Hit_By_Pitch+a.Sac_Fly),0),0) as OBP
FROM PitchGameCounts as a
JOIN PitchGameCounts as rolling_stat
ON a.team_id  = rolling_stat.team_id 
	AND a.local_date > rolling_stat.local_date 
	AND rolling_stat.local_date BETWEEN a.local_date - INTERVAL 100 DAY 
	AND a.local_date
GROUP BY rolling_stat.team_id , rolling_stat.game_id
ORDER BY rolling_stat.local_date DESC, rolling_stat.game_id DESC, rolling_stat.team_id DESC;

delete from Master where winner_home_or_away =''; 

SELECT * from Master m ; 





