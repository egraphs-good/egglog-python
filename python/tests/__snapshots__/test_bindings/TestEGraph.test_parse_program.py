[
    Datatype(
        EgglogSpan(
            SrcFile(
                "test.egg",
                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
            ),
            0,
            110,
        ),
        "Math",
        [
            Variant(
                EgglogSpan(
                    SrcFile(
                        "test.egg",
                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                    ),
                    25,
                    34,
                ),
                "Num",
                ["i64"],
                None,
            ),
            Variant(
                EgglogSpan(
                    SrcFile(
                        "test.egg",
                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                    ),
                    45,
                    57,
                ),
                "Var",
                ["String"],
                None,
            ),
            Variant(
                EgglogSpan(
                    SrcFile(
                        "test.egg",
                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                    ),
                    68,
                    83,
                ),
                "Add",
                ["Math", "Math"],
                None,
            ),
            Variant(
                EgglogSpan(
                    SrcFile(
                        "test.egg",
                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                    ),
                    94,
                    109,
                ),
                "Mul",
                ["Math", "Math"],
                None,
            ),
        ],
    ),
    ActionCommand(
        Let(
            EgglogSpan(
                SrcFile(
                    "test.egg",
                    '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                ),
                151,
                200,
            ),
            "expr1",
            Call(
                EgglogSpan(
                    SrcFile(
                        "test.egg",
                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                    ),
                    162,
                    199,
                ),
                "Mul",
                [
                    Call(
                        EgglogSpan(
                            SrcFile(
                                "test.egg",
                                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                            ),
                            167,
                            174,
                        ),
                        "Num",
                        [
                            Lit(
                                EgglogSpan(
                                    SrcFile(
                                        "test.egg",
                                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                    ),
                                    172,
                                    173,
                                ),
                                Int(2),
                            )
                        ],
                    ),
                    Call(
                        EgglogSpan(
                            SrcFile(
                                "test.egg",
                                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                            ),
                            175,
                            198,
                        ),
                        "Add",
                        [
                            Call(
                                EgglogSpan(
                                    SrcFile(
                                        "test.egg",
                                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                    ),
                                    180,
                                    189,
                                ),
                                "Var",
                                [
                                    Lit(
                                        EgglogSpan(
                                            SrcFile(
                                                "test.egg",
                                                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                            ),
                                            185,
                                            188,
                                        ),
                                        String("x"),
                                    )
                                ],
                            ),
                            Call(
                                EgglogSpan(
                                    SrcFile(
                                        "test.egg",
                                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                    ),
                                    190,
                                    197,
                                ),
                                "Num",
                                [
                                    Lit(
                                        EgglogSpan(
                                            SrcFile(
                                                "test.egg",
                                                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                            ),
                                            195,
                                            196,
                                        ),
                                        Int(3),
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )
    ),
]