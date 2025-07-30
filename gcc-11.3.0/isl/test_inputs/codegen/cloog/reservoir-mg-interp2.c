{
  for (int c0 = 1; c0 < O; c0 += 1)
    for (int c1 = Q; c1 < N; c1 += 1)
      for (int c2 = P; c2 < M; c2 += 1)
        S1(c0, c1, c2);
  for (int c0 = 1; c0 < O; c0 += 1)
    for (int c1 = Q; c1 < N; c1 += 1)
      for (int c2 = 1; c2 < M; c2 += 1)
        S2(c0, c1, c2);
  for (int c0 = 1; c0 < O; c0 += 1)
    for (int c1 = 1; c1 < N; c1 += 1)
      for (int c2 = P; c2 < M; c2 += 1)
        S3(c0, c1, c2);
  for (int c0 = 1; c0 < O; c0 += 1)
    for (int c1 = 1; c1 < N; c1 += 1)
      for (int c2 = 1; c2 < M; c2 += 1)
        S4(c0, c1, c2);
}
