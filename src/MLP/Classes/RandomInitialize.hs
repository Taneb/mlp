{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE QuantifiedConstraints #-}
module MLP.Classes.RandomInitialize (RandomInitialize (..)) where

import Control.Applicative
import Control.Category
import Control.Monad.Primitive
import System.Random.MWC

import Prelude hiding (id, (.))

import MLP.Core

class RandomInitialize l where
  randomInitialize :: PrimMonad m => l i o -> Gen (PrimState m) -> m (l i o)

newtype RI m c i o = RI {runRI :: Gen (PrimState m) -> m (Network c i o)}

instance Applicative m => Category (RI m c) where
  id = RI $ \_ -> pure id
  RI f . RI g = RI $ (liftA2 . liftA2) (.) f g

instance (forall l. c l => RandomInitialize l) => RandomInitialize (Network c) where
  randomInitialize = runRI . free (RI . fmap (fmap forget) . randomInitialize)
