for (int c0 = 5; c0 <= 5 * M; c0 += 1) {
  for (int c1 = max(2, floord(-M + c0, 4)); c1 < min(-((5 * M - c0 + 1) % 2) + M, (c0 + 1) / 3 - 2); c1 += 1)
    for (int c2 = max(1, -M - c1 + (M + c0) / 2 - 2); c2 < min(c1, -2 * c1 + (c0 + c1) / 2 - 2); c2 += 1)
      S1(c0 - 2 * c1 - 2 * c2 - 5, c1, c2);
  for (int c1 = max(1, floord(-M + c0, 4)); c1 < (c0 + 1) / 5; c1 += 1)
    S2(c0 - 4 * c1 - 3, c1);
  if (c0 % 5 == 0)
    S4(c0 / 5);
  for (int c1 = max(-3 * M - c0 + 3 * ((M + c0) / 2) + 1, -((c0 - 1) % 3) + 3); c1 < (c0 + 1) / 5; c1 += 3)
    S3((c0 - 2 * c1 - 1) / 3, c1);
}
