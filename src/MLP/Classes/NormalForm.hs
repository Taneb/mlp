{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE UndecidableInstances #-}
module MLP.Classes.NormalForm (NormalFormNetwork(..)) where

import Control.Category

import MLP.Core

class NormalFormNetwork l where
  normalFormNetwork :: l i o -> ()

data NFN i o = NFN

instance Category NFN where
  id = NFN
  NFN . NFN = NFN

instance (forall l. c l => NormalFormNetwork l) => NormalFormNetwork (Network c) where
  normalFormNetwork net = case free (\l -> normalFormNetwork l `seq` NFN) net of
    NFN -> ()

