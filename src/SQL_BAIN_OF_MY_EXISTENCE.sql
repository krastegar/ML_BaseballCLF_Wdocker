USE baseball; 

SHOW TABLES;


-- Making My Master Table
DROP TABLE IF EXISTS Master; 
CREATE TABLE Master
SELECT bc.game_id, batter, atBat, Hit, team_id, local_date
FROM batter_counts bc 
JOIN game g 
ON bc.game_id = g.game_id; 
-- Index for Master Table for faster query
ALTER TABLE Master
ADD ID INT NOT NULL AUTO_INCREMENT, 
ADD CONSTRAINT PRIMARY KEY(ID); 

-- Historic AVG
DROP TABLE IF EXISTS Hist_Avg;
CREATE TABLE Hist_Avg
SELECT batter, SUM(Hit)/NULLIF (SUM(atBat),0) as Batting_Avg
FROM Maste
GROUP BY batter; 

-- Annual Avg
DROP TABLE IF EXISTS Annual_Avg;
CREATE TABLE Hist_Avg
SELECT batter, SUM(Hit)/NULLIF (SUM(atBat),0) as Batting_Avg,
YEAR (local_date) as year
FROM Master
GROUP BY batter, year; 

-- Beginning of Rolling AVG

-- Rolling Average
/*DROP TABLE IF EXISTS Rolling_Avg;
CREATE TABLE Rolling_Avg
SELECT
       a.local_date,
       a.atBat,
       a.Hit,
       a.batter,
       Round( ( SELECT SUM(b.Hit)/NULLIF (SUM(b.atBat),0)
                FROM Master AS b
                WHERE DATEDIFF(a.local_date, b.local_date) BETWEEN 0 AND 99
              ), 2 ) AS '100dayMovingAvg'
     FROM Master AS a
     GROUP BY batter
     ORDER BY a.local_date DESC;

*/

-- Intermediate Temp table 
DROP TEMPORARY TABLE IF EXISTS temp; 
CREATE TEMPORARY TABLE temp
SELECT bc.game_id, bc.batter, atBat, Hit, team_id, local_date,
STR_TO_DATE(g.local_date , "%Y-%m-%d %H:%i:%s")
FROM batter_counts bc 
JOIN game g 
ON bc.game_id = g.game_id; 

-- Self Join my final rol_avg
DROP TABLE IF EXISTS Rol_AVG;
CREATE TABLE Rol_AVG
SELECT t.atBat, t.Hit, t.batter,
SUM(t.Hit)/NULLIF (SUM(t.atBat),0) as average,
STR_TO_DATE(t.local_date , "%Y-%m-%d %H:%i:%s")
FROM temp as t 
JOIN temp as rolling_avg
ON t.batter = rolling_avg.batter 
	AND t.local_date > rolling_avg.local_date 
	AND rolling_avg.local_date BETWEEN t.local_date - INTERVAL 100 DAY 
	AND t.local_date
GROUP BY rolling_avg.batter
ORDER BY rolling_avg.local_date DESC;

-- a.batter = b.batter and a.local_date > b.local_date and b.local_date between  a.local_date - INTERVAL 100 DAY and a.local_date
-- BETWEEN DATE_ADD(t.local_date, INTERVAL -99 DAY) and t.local_date 