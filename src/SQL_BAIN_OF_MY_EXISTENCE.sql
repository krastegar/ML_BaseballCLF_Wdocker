USE baseball; 

SHOW TABLES;


-- Making My Master Table
DROP TABLE IF EXISTS Master; 
CREATE TABLE Master
SELECT bc.game_id, batter, atBat, Hit, team_id, local_date
FROM batter_counts bc 
JOIN game g 
ON bc.game_id = g.game_id; 

ALTER TABLE Master
ADD ID INT NOT NULL AUTO_INCREMENT, 
ADD CONSTRAINT PRIMARY KEY(ID); 

-- Historic AVG
DROP TABLE IF EXISTS Hist_Avg;
CREATE TABLE Hist_Avg
SELECT batter, SUM(Hit)/NULLIF (SUM(atBat),0) as Batting_Avg
FROM Master
GROUP BY batter; 

-- Annual Avg
DROP TABLE IF EXISTS Annual_Avg;
CREATE TABLE Hist_Avg
SELECT batter, SUM(Hit)/NULLIF (SUM(atBat),0) as Batting_Avg,
YEAR (local_date) as year
FROM Master
GROUP BY batter, year; 

-- Beginning of Rolling AVG
SELECT batter, SUM(Hit)/NULLIF (SUM(atBat),0) as Batting_Avg,
YEAR (local_date) as year
FROM Master
WHERE Master.local_date > ( CURDATE() - INTERVAL 100 DAY )
GROUP BY batter, Master.local_date; 


