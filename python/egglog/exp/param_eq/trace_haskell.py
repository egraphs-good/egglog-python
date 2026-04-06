"""Stepwise Haskell tracing for the retained param-eq replication cases."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from egglog.exp.param_eq.generate_haskell_golden import _canonicalize, _parse_analysis, _source_to_haskell_expr
from egglog.exp.param_eq.paths import llvm_bin_dir, param_eq_data_dir
from egglog.exp.param_eq.trace_egglog import TRACE_ROOT, TraceResult
from egglog.exp.param_eq.trace_tables import JsonValue, SnapshotTables

HASKELL_ROOT = param_eq_data_dir()


def haskell_trace_available() -> bool:
    """Return whether the local Haskell tracing prerequisites are present."""
    return HASKELL_ROOT.exists() and shutil.which("stack") is not None


def _build_haskell_trace_program(source: str, *, outer_pass: int) -> str:
    outer_prefix = f"outer_{outer_pass}"
    return "\n".join([
        "{-# LANGUAGE BangPatterns #-}",
        "{-# LANGUAGE BlockArguments #-}",
        "{-# LANGUAGE FlexibleContexts #-}",
        "{-# LANGUAGE LambdaCase #-}",
        "{-# LANGUAGE ScopedTypeVariables #-}",
        "{-# LANGUAGE TupleSections #-}",
        "import Control.Monad ((<=<), forM_, when)",
        "import Data.Bifunctor (first)",
        "import qualified Data.IntMap.Strict as IM",
        "import qualified Data.Set as S",
        "import Data.List (intercalate)",
        "import Data.SRTree",
        "import Data.SRTree.Print",
        "import FixTree",
        "import qualified Data.Equality.Graph as G",
        "import qualified Data.Equality.Graph.Lens as L",
        "import Data.Equality.Graph.Lens hiding ((^.))",
        "import Data.Equality.Graph.Monad",
        "import Data.Equality.Graph.Nodes",
        "import Data.Equality.Matching",
        "import Data.Equality.Matching.Database",
        "import Data.Equality.Extraction",
        "import Data.Equality.Language",
        "import Data.Equality.Saturation (Fix(..), Rewrite(..))",
        "import Data.Equality.Saturation.Scheduler",
        "",
        f'outerPrefix :: String\nouterPrefix = "{outer_prefix}"',
        "",
        "sanitize :: String -> String",
        "sanitize = map (\\c -> if c == '\\t' || c == '\\n' then ' ' else c)",
        "",
        "joinFields :: [String] -> String",
        'joinFields = intercalate "\\t" . map sanitize',
        "",
        "joinChildren :: [String] -> String",
        'joinChildren = intercalate "|" . map sanitize',
        "",
        "emitSnapshot :: String -> Int -> G.EGraph SRTreeF -> [String]",
        "emitSnapshot step root egr =",
        "  let rootClass = G.find root egr",
        "      rootData = show (egr L.^. _class rootClass._data)",
        "      rootExtract = showDefault . relabelParams . toSRTree $ extractBest egr cost2 rootClass",
        "      classPairs = IM.toAscList (egr L.^. _classes)",
        "      nodeCount = sum [S.size (cls L.^. _nodes) | (_, cls) <- classPairs]",
        "      memoCount = sizeNM (egr L.^. _memo)",
        '      metaLine = joinFields ["META", step, show rootClass, rootData, show (length classPairs), show nodeCount, show memoCount]',
        '      rootLine = joinFields ["ROOT", step, show rootClass, rootData, rootExtract]',
        "      classLines =",
        '        [ joinFields ["CLASS", step, show cid, show (cls L.^. _data), showDefault . relabelParams . toSRTree $ extractBest egr cost2 cid, show (S.size (cls L.^. _nodes))]',
        "        | (cid, cls) <- classPairs",
        "        ]",
        "      nodeLines =",
        '        [ joinFields ["NODE", step, show cid, show node, show (operator node), joinChildren (map show (children node))]',
        "        | (cid, cls) <- classPairs",
        "        , node <- S.toList (cls L.^. _nodes)",
        "        ]",
        '      memoLines = foldrWithKeyNM\' (\\node cid acc -> joinFields ["MEMO", step, show node, show cid] : acc) [] (egr L.^. _memo)',
        "   in metaLine : rootLine : classLines <> nodeLines <> memoLines",
        "",
        "scheduler :: BackoffScheduler",
        "scheduler = BackoffScheduler 2500 30",
        "",
        "rewrites :: [Rewrite SRTreeF]",
        "rewrites = rewritesBasic <> rewritesFun",
        "",
        "matchWithScheduler :: Database SRTreeF -> Int -> IM.IntMap (Stat BackoffScheduler) -> (Int, Rewrite SRTreeF) -> ([(Rewrite SRTreeF, Match)], IM.IntMap (Stat BackoffScheduler))",
        "matchWithScheduler db i stats = \\case",
        "  (rwId, rw :| cnd) -> first (map (first (:| cnd))) $ matchWithScheduler db i stats (rwId, rw)",
        "  (rwId, lhs := rhs) ->",
        "    case IM.lookup rwId stats of",
        "      Just s | isBanned i s -> ([], stats)",
        "      x ->",
        "        let matches' = ematch db lhs",
        "            newStats = updateStats scheduler i rwId x stats matches'",
        "         in (map (lhs := rhs,) matches', newStats)",
        "",
        "applyMatchesRhs :: (Rewrite SRTreeF, Match) -> EGraphM SRTreeF ()",
        "applyMatchesRhs = \\case",
        "  (rw :| cond, m@(Match subst _)) -> do",
        "    egr <- get",
        "    when (cond subst egr) $ applyMatchesRhs (rw, m)",
        "  (_ := VariablePattern v, Match subst eclass) ->",
        "    case IM.lookup v subst of",
        '      Nothing -> error "impossible: couldn\'t find v in subst"',
        "      Just n -> do",
        "        _ <- merge n eclass",
        "        pure ()",
        "  (_ := NonVariablePattern rhs, Match subst eclass) -> do",
        "    eclass' <- reprPat subst rhs",
        "    _ <- merge eclass eclass'",
        "    pure ()",
        "",
        "reprPat :: Subst -> SRTreeF (Pattern SRTreeF) -> EGraphM SRTreeF Int",
        "reprPat subst = add . Node <=< traverse \\case",
        "  VariablePattern v ->",
        "    case IM.lookup v subst of",
        '      Nothing -> error "impossible: couldn\'t find v in subst?"',
        "      Just i -> pure i",
        "  NonVariablePattern p -> reprPat subst p",
        "",
        "traceLoop :: Int -> Int -> IM.IntMap (Stat BackoffScheduler) -> EGraphM SRTreeF [String]",
        "traceLoop root i stats",
        "  | i > 30 = pure []",
        "  | otherwise = do",
        "      egr <- get",
        "      let (beforeMemo, beforeClasses) = (egr L.^. _memo, egr L.^. _classes)",
        "          db = eGraphToDatabase egr",
        "          (!matches, newStats) = mconcat (fmap (matchWithScheduler db i stats) (zip [1..] rewrites))",
        "      forM_ matches applyMatchesRhs",
        "      rebuild",
        '      snapshot <- gets (emitSnapshot (outerPrefix <> "_inner_" <> show i <> "_after_rebuild") root)',
        "      (afterMemo, afterClasses) <- gets (\\g -> (g L.^. _memo, g L.^. _classes))",
        "      if sizeNM afterMemo == sizeNM beforeMemo && IM.size afterClasses == IM.size beforeClasses",
        "        then pure snapshot",
        "        else do",
        "          rest <- traceLoop root (i + 1) newStats",
        "          pure (snapshot <> rest)",
        "",
        "main :: IO ()",
        "main = do",
        f"  let expr = toFixTree ({_source_to_haskell_expr(source)})",
        "      ((_, rows), _) = egraph $ do",
        "        root <- represent expr",
        '        startRows <- gets (emitSnapshot (outerPrefix <> "_pass_start") root)',
        "        loopRows <- traceLoop root 1 mempty",
        '        extractRows <- gets (emitSnapshot (outerPrefix <> "_extract") root)',
        "        pure (root, startRows <> loopRows <> extractRows)",
        "  mapM_ putStrLn rows",
        "",
    ])


def _run_haskell_trace_pass(source: str, *, outer_pass: int) -> str:
    program = _build_haskell_trace_program(source, outer_pass=outer_pass)
    with tempfile.NamedTemporaryFile("w", suffix=".hs", delete=False) as handle:
        handle.write(program)
        temp_path = Path(handle.name)
    try:
        stack = shutil.which("stack")
        if stack is None:
            msg = "stack is required for Haskell tracing"
            raise RuntimeError(msg)
        env = dict(os.environ)
        llvm_bin = llvm_bin_dir()
        if llvm_bin is not None:
            env["PATH"] = f"{llvm_bin}:{env['PATH']}"
        return subprocess.check_output(
            [stack, "exec", "--", "runghc", "-isrc", str(temp_path)],
            cwd=HASKELL_ROOT,
            env=env,
            text=True,
            timeout=300,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _rows_to_snapshots(
    output: str,
    *,
    case_id: str,
    source: str,
) -> dict[str, SnapshotTables]:
    grouped: dict[str, dict[str, list[dict[str, JsonValue]]]] = {}
    metadata_by_step: dict[str, dict[str, JsonValue]] = {}
    for raw_line in output.splitlines():
        record = raw_line.split("\t")
        kind, step = record[0], record[1]
        grouped.setdefault(step, {"classes": [], "nodes": [], "memo": [], "root": [], "functions": [], "rows": []})
        metadata = metadata_by_step.setdefault(
            step,
            {
                "system": "haskell",
                "case_id": case_id,
                "step": step,
                "source": source,
            },
        )
        if kind == "META":
            metadata["root_class_id"] = record[2]
            metadata["root_analysis"] = _parse_analysis(record[3])
            metadata["class_count"] = int(record[4])
            metadata["node_count"] = int(record[5])
            metadata["memo_size"] = int(record[6])
            continue
        if kind == "ROOT":
            root_expr = _canonicalize(record[4])
            analysis = _parse_analysis(record[3])
            metadata["root_class_id"] = record[2]
            metadata["root_analysis"] = analysis
            metadata["root_extracted_expr"] = root_expr
            grouped[step]["root"].append({
                "root_class_id": record[2],
                "root_analysis": analysis,
                "extracted_expr": root_expr,
            })
            continue
        if kind == "CLASS":
            grouped[step]["classes"].append({
                "class_id": record[2],
                "analysis": _parse_analysis(record[3]),
                "best_expr": _canonicalize(record[4]),
                "node_count": int(record[5]),
            })
            continue
        if kind == "NODE":
            grouped[step]["nodes"].append({
                "class_id": record[2],
                "node_repr": record[3],
                "op": record[4],
                "children": [] if not record[5] else record[5].split("|"),
            })
            continue
        if kind == "MEMO":
            grouped[step]["memo"].append({
                "node_repr": record[2],
                "class_id": record[3],
            })
            continue
        msg = f"Unexpected Haskell trace line: {raw_line!r}"
        raise ValueError(msg)
    return {step: SnapshotTables(metadata=metadata_by_step[step], tables=tables) for step, tables in grouped.items()}


def trace_haskell_case(
    *,
    case_id: str,
    source: str,
    output_root: Path = TRACE_ROOT,
) -> TraceResult:
    """Trace Haskell `rewriteTree`/`simplifyE` one pass at a time."""
    if not haskell_trace_available():
        msg = "Haskell tracing requires stack and a checked-out param-eq-haskell tree"
        raise RuntimeError(msg)

    output_dir = output_root / case_id / "haskell"
    output_dir.mkdir(parents=True, exist_ok=True)
    step_paths: list[Path] = []
    current_source = source
    final_rendered = source
    for outer_pass in range(1, 3):
        snapshots = _rows_to_snapshots(
            _run_haskell_trace_pass(current_source, outer_pass=outer_pass),
            case_id=case_id,
            source=source,
        )
        for step_name, snapshot in snapshots.items():
            step_path = output_dir / f"{step_name}.json"
            snapshot.write_json(step_path)
            step_paths.append(step_path)
        extract_snapshot = snapshots[f"outer_{outer_pass}_extract"]
        extract_rows = extract_snapshot.tables["root"]
        assert extract_rows
        final_rendered = str(extract_rows[0]["extracted_expr"])
        if final_rendered == _canonicalize(current_source):
            current_source = final_rendered
            break
        current_source = final_rendered

    final_snapshot = SnapshotTables(
        metadata={
            "system": "haskell",
            "case_id": case_id,
            "step": "final_simplify_e",
            "phase": "extraction",
            "source": source,
            "final_rendered": final_rendered,
        },
        tables={
            "root": [
                {
                    "extracted_expr": final_rendered,
                }
            ],
        },
    )
    final_path = output_dir / "final_simplify_e.json"
    final_snapshot.write_json(final_path)
    step_paths.append(final_path)
    return TraceResult(
        system="haskell",
        case_id=case_id,
        output_dir=output_dir,
        step_paths=step_paths,
        final_rendered=final_rendered,
    )
