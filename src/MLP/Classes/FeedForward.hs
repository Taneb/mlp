{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE UndecidableInstances #-}
module MLP.Classes.FeedForward (FeedForwardCPU (..)) where

import Control.Category

import Prelude hiding ((.), id)

import MLP.Core

class FeedForwardCPU l where
  feedForwardCPU :: l i o -> TensorTree i -> TensorTree o

newtype FFCPU i o = FFCPU {runFFCPU :: TensorTree i -> TensorTree o}

instance Category FFCPU where
  id = FFCPU id
  FFCPU x . FFCPU y = FFCPU (x . y)

instance (forall l. c l => FeedForwardCPU l) => FeedForwardCPU (Network c) where
  feedForwardCPU = runFFCPU . free (FFCPU . feedForwardCPU)
