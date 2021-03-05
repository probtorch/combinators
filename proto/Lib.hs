-- | A peek behind the curtain

{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoImplicitPrelude #-}
module Lib where

import Prelude hiding (lookup)
import Control.Lens
import Control.Monad.Free
import Control.Monad.State
import Text.Show.Deriving
import Data.HashMap.Strict
import Control.Monad.ST
import System.Random.MWC.Probability hiding (sample)
import qualified Data.HashMap.Strict as HM
import qualified System.Random.MWC.Probability as P

-- some small distributions to work with
data Dist
  = Multinomial Int [Double]
  | Normal Double Double
  deriving (Eq, Show)

type Address = String

-- PPL functionality (-F for "functor"). Functor means:
--
--     instance Functor PPLF where
--       fmap f (Sample  name dist k)        = Sample  name dist (f . k)
--       fmap f (Observe name dist obs next) = Observe name dist obs (f next)
--       fmap _ (End xs) = End xs
data PPLF next
  = Sample  Address Dist ([Double] -> next)
  | Observe Address Dist  [Double]    next
--  | End [Double]
  deriving Functor

type PPL = Free PPLF

sample :: Address -> Dist -> PPL [Double]
sample n d = liftF (Sample n d id)

observe :: Address -> Dist -> [Double] -> PPL ()
observe n d obs = liftF (Observe n d obs ())

-- end :: [Double] -> PPL [Double]
-- end xs = liftF (End xs)

-- how to run a handler
stateHandler :: PPL ~> State (Seed, HashMap Address (Either Dist [Double]))
stateHandler (Pure x) = pure x
stateHandler (Free (Sample a d f)) = do
  (s, tr) <- get
  case lookup a tr of
    Nothing -> do
      let xs = runST $ runDist s d
      put (s, HM.insert a (Right xs) tr)
      stateHandler (f xs)
    Just (Right vals) -> stateHandler (f vals)
    Just (Left d) -> do
      seed <- view _1 <$> get
      let vals = runST (runDist seed d)
      put (s, HM.insert a (Right vals) tr)
      stateHandler (f vals)

stateHandler (Free (Observe a d o f)) = do
  (s, tr) <- get
  case lookup a tr of
    -- then we add it because we don't really care too much about correctness
    Nothing -> put (s, HM.insert a (Right $ runST $ runDist s d) tr) >> stateHandler f
    Just  _ -> put (s, HM.insert a (Right o) tr) >> stateHandler f


runDist :: Seed -> Dist -> ST s [Double]
runDist seed d = do
  gen <- restore seed
  sampleit d gen
  where
    sampleit :: forall s . Dist -> GenST s -> ST s [Double]
    sampleit (Normal μ σ) g       = (:[]) <$> P.sample (normal μ σ) g
    sampleit (Multinomial i ps) g = fromIntegral <$$> P.sample (multinomial i ps) g


-- * Misc

-- | showing things
-- TODO: switch this with a seed value
-- TODO: switch to Show1
showPPL :: Show a => PPL a -> String
showPPL (Pure a)                 = "return " ++ show a
showPPL (Free (Sample a d k))    = "_ <~ sample " ++ show a ++ " (" ++ show d ++ ")\n" ++ showPPL (k [0]) -- just thread a number through for now
showPPL (Free (Observe a d o k)) = "observe " ++ show a ++ " (" ++ show d ++ ") " ++ show o ++ "\n" ++ showPPL k

-- | printing
printPPL :: Show a => PPL a -> IO ()
printPPL = Prelude.putStrLn . showPPL

-- | A natural transformation
infixr 0 ~>
type f ~> g = forall x. f x -> g x

(<$$>) :: Functor f => Functor g => (a -> b) -> f (g a) -> f (g b)
(<$$>) = fmap.fmap

-- | Console functionality.
data ConsoleF a
  = PutStrLn String a
  deriving (Functor)
$(deriveShow1 ''ConsoleF)

type Console = Free ConsoleF

putStrLn :: String -> Console ()
putStrLn s = liftF (PutStrLn s ())

consoleIO :: ConsoleF ~> IO
consoleIO (PutStrLn s v) = do
  Prelude.putStrLn s
  pure v

