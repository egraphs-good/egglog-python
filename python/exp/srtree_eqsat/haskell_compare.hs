{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE LambdaCase #-}

module Main where

import Data.SRTree
import Data.SRTree.EqSat (simplifyEqSat)
import Data.SRTree.Print (showPython)
import System.CPUTime (getCPUTime)
import System.Environment (getArgs)
import System.Exit (die)

data RunResult = RunResult
  { rowIndex :: Int
  , runtimeSec :: Double
  , stopReason :: String
  , beforeParams :: Int
  , afterParams :: Int
  , beforeNodeCount :: Int
  , afterNodeCount :: Int
  , finalMemoSize :: Maybe Int
  , finalEClassCount :: Maybe Int
  , simplifiedPython :: String
  , notes :: [String]
  }

main :: IO ()
main = do
  args <- getArgs
  let rows = if null args then [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200] else map read args
  mapM_ runRow rows

runRow :: Int -> IO ()
runRow row =
  case selectedRowExpr row of
    Nothing -> die $ "helper currently supports rows 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, and 200 only, got " <> show row
    Just tree -> do
      start <- getCPUTime
      let simplified = simplifyEqSat tree
          simplifiedPy = showPython simplified
      simplifiedPy `seq` pure ()
      end <- getCPUTime
      let elapsed = fromIntegral (end - start) / 1e12 :: Double
          result =
            RunResult
              { rowIndex = row
              , runtimeSec = elapsed
              , stopReason = "unavailable_from_exported_api"
              , beforeParams = countParams . fst $ floatConstsToParam tree
              , afterParams = countParams . fst $ floatConstsToParam simplified
              , beforeNodeCount = countNodes tree
              , afterNodeCount = countNodes simplified
              , finalMemoSize = Nothing
              , finalEClassCount = Nothing
              , simplifiedPython = simplifiedPy
              , notes =
                  [ "This helper uses the source repo's exported simplifyEqSat."
                  , "The public API does not expose intermediate e-graph sizes or stop reasons."
                  ]
              }
      printRow result

printRow :: RunResult -> IO ()
printRow res = do
  putStrLn $ "row\t" <> show (rowIndex res)
  putStrLn $ "runtime_sec\t" <> show (runtimeSec res)
  putStrLn $ "stop_reason\t" <> stopReason res
  putStrLn $ "before_params\t" <> show (beforeParams res)
  putStrLn $ "after_params\t" <> show (afterParams res)
  putStrLn $ "before_node_count\t" <> show (beforeNodeCount res)
  putStrLn $ "after_node_count\t" <> show (afterNodeCount res)
  putStrLn $ "final_memo_size\t" <> maybe "na" show (finalMemoSize res)
  putStrLn $ "final_eclass_count\t" <> maybe "na" show (finalEClassCount res)
  mapM_ (\note -> putStrLn $ "note\t" <> note) (notes res)
  putStrLn $ "simplified_python\t" <> simplifiedPython res
  putStrLn "---"

selectedRowExpr :: Int -> Maybe (Fix SRTree)
selectedRowExpr = \case
  1 -> Just row1Expr
  2 -> Just row2Expr
  3 -> Just row3Expr
  4 -> Just row4Expr
  5 -> Just row5Expr
  10 -> Just row10Expr
  20 -> Just row20Expr
  30 -> Just row30Expr
  40 -> Just row40Expr
  50 -> Just row50Expr
  75 -> Just row75Expr
  100 -> Just row100Expr
  150 -> Just row150Expr
  200 -> Just row200Expr
  _ -> Nothing

row1Expr :: Fix SRTree
row1Expr = sqr (-9.29438919215253 + 2.93547417364396 * theta_)

row2Expr :: Fix SRTree
row2Expr =
  ( (((2.91970418475328 + (-0.773583225885789) * theta_) * 2.87230609775115 * theta_)
        + ((-1.22356797479824 + 0.377885426397629 * theta_) * (10.3917281470804 - ((-1.20862070725184) * alpha_)))
    )
      * (0.114247677264604 * alpha_ + 0.127799885276295 * beta_)
      * ((3.60250727355285 * beta_ + 1.35381696253641 * theta_) - 9.60356246300082 * alpha_)
      - ((sqr (-9.29438919215253 + 2.93547417364396 * theta_) + (((-1.31893797108115) * beta_) + 2.10814063839763 * alpha_))
            - 5.16995214427275 * alpha_
        )
          * (1.25099225339683 * alpha_ + (-1.0075335593301) * beta_)
    )
    * 0.227993040492379
    + 1.72695871415228

row3Expr :: Fix SRTree
row3Expr =
  ( (((2.78948361720301 + (-0.603311011990701) * theta_) * 2.66295898187985 * theta_)
        + ((-1.70053796438431 + 0.354410284952137 * theta_) * (2.30920195902924 * alpha_ + (-0.684514001199842) * beta_))
    )
      * (((-0.423113013561051) * alpha_) + ((-0.820293590903207) * beta_))
      * exp (0.648433111594964 * alpha_)
      - ((((4.84782777418965 * alpha_) + ((-10.2013161038227) * beta_)) + (4.49181352322002 + 3.34338756313595 * theta_))
            - 1.52463076043089 * alpha_
        )
          * sqr (-4.40371470183112 + 1.239832234323 * theta_)
    )
    * 5.55952399242293e-2
    + 2.3974618452108

row4Expr :: Fix SRTree
row4Expr =
  ( (((2.84721301451649 + (-1.0832466055762) * theta_) * 2.87288386010019 * theta_)
        + ((-0.960822430613027 + 0.352212452215726 * theta_) * (9.49502848514098 - ((-4.43457932618305) * alpha_)))
    )
      * (0.0231254136991372 * alpha_ + (-0.201859225439622) * beta_)
      * exp (0.533684665488698 * alpha_)
      - ((((4.3709278998084 * alpha_) + 6.76271161563555e-2 * beta_) + ((-10.1636621343434) + 2.20402481737344 * theta_))
            - 2.00153063481211 * alpha_
        )
          * (((-0.173514070270718) * alpha_) + 1.4309773332239 * beta_)
    )
    * 0.418278900765285
    + (-0.250852384709946)

row5Expr :: Fix SRTree
row5Expr =
  ( 2.98279150112649
      - 2.50335104760053 * alpha_
      + exp (0.472180751947384 * alpha_)
      + (((( (-5.46511983933291) * alpha_) + 4.7308388959051 * beta_) + sqr (1.9820027542127 * alpha_ + (-0.507993116540432) * beta_))
            + ((-0.902691894725103) * theta_)
        )
          * (sqr (-9.71025706279119 + 3.06939441276146 * theta_) + (-12.1940251737044))
          * (((-1.13611973003391e-3) * alpha_) + 6.9224336526928e-3 * beta_)
    )
    * 1.38844130555412
    + 0.330982870226231

row10Expr :: Fix SRTree
row10Expr =
  ( ((-4.0257291259403) * alpha_) + exp (0.496521530470785 * alpha_) * (sqr (-9.90385501742711 + 3.09891553397754 * theta_) + (-12.0313313266527)) * (((-2.09790347331461e-2) * alpha_) + 0.15542475900314 * beta_)
  )
    * 0.218016003919151
    + 2.93075744602469

row20Expr :: Fix SRTree
row20Expr = sqr (cube ((-0.453946451098545) * theta_) + 3.0472620183122 * theta_) * (-0.342063915063894) + 10.6692145111104

row30Expr :: Fix SRTree
row30Expr =
  ( exp ((-1.47422568429434) * alpha_) - (cube ((-0.58923877562757) * theta_) + sqr (2.39236882178653 * theta_ - 6.26125635041365)) * ((-7.92212718106198e-3) * beta_)
  )
    * (exp (1.19822491431359 * alpha_) + 19.3608807319069 * beta_)
    * 0.313237693440143
    + (-0.504509880559088)

row40Expr :: Fix SRTree
row40Expr = sqr (0.309853513321887 * theta_ - exp ((-4.78572314032874e-3) * alpha_)) * 32.3824199084528 + (-4.32199869886884)

row50Expr :: Fix SRTree
row50Expr =
  ( exp (0.743694003014863 * alpha_)
      * ((-0.0121179632900701 * theta_) + (0.00904122619609017 * alpha_))
      * ((-3.05659895630567 * theta_) + 8.63005732191704)
      + (-0.557193153898209 * alpha_)
      - log (0.782997897866162 * theta_)
      + sqr (exp (-0.144728813168975 * theta_)) * (-1.54770141702422 + (-3.31046821812388 * theta_))
      + 6.34281372899835
  )
    * 1.70035591779884
    * beta_
    * 1.03068155805492
    + 0.404362453868565

row75Expr :: Fix SRTree
row75Expr =
  ( 0.27658499902994 * beta_
      + ((((-3.09568685791928) * theta_ - 1.06543349470034 * beta_) + 4.80847033528277 * alpha_)
          * (0.00840515601044034 * theta_ - exp (((-5.89563245316599) * theta_) - ((-2.2128445010022) * beta_)))
        )
  )
    * (sqr ((-1.73329369062705) * theta_ + 5.01701830214846) - 3.04752460994615 * alpha_)
    * 0.830662588380978
    + 2.77768079307633

row100Expr :: Fix SRTree
row100Expr =
  (((-0.817246761082895) * theta_) + 4.7987094782841 - 1.18369793731582 * alpha_ * (3.27838657131292 * theta_ + (-15.2760394159237)) * (((-9.53755961420584e-2) * theta_) + 0.0270703912792823 * alpha_))
    * 1.11985975437354
    * beta_
    * 1.01334538331164
    + 0.201404917055389

row150Expr :: Fix SRTree
row150Expr =
  ( (((5.53327625349294 * beta_ - 9.01964233670581 * theta_) - (((-17.4546811325902) * alpha_) - 2.50583781801447 * alpha_))
        * (((-2.49821516165647)) - ((-0.402138004942828) * theta_))
        * (log (0.347653612616773 * alpha_) - ((-0.512350969799209) * theta_))
      )
      - (((1.40058250223625 * alpha_) - ((-0.313213248196185) * beta_) - ((-0.244937726279955) * beta_))
          * exp (0.569380315263419 * alpha_)
          * (((-0.841465262470357) * theta_) + exp (0.242251869587583 * alpha_) - sqr (0.466508892559209 * theta_ + (-2.80204970690573)))
        )
  )
    * 6.32264979538498e-2
    * beta_
    * 1.00057660932339
    + (-0.455923813201509)

row200Expr :: Fix SRTree
row200Expr =
  (sqr (((-7.01398906858873) * theta_ - ((-20.9157503480256)) + 0.40965685006749 * beta_)) - 18.7131049568053 * alpha_)
    / (((-2.89176232552097) * beta_) + 15.8269215595989 + sqr (((-1.76570724282939) * theta_) - ((-5.46414193899484))))
    * 0.689375955248431
    + 0.570693365253855

alpha_, beta_, theta_ :: Fix SRTree
alpha_ = var 0
beta_ = var 1
theta_ = var 2

sqr :: Fix SRTree -> Fix SRTree
sqr x = x ** 2

cube :: Fix SRTree -> Fix SRTree
cube x = x ** 3
