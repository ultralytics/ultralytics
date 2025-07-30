if (m <= 1) {
  for (int c0 = 1; c0 <= n; c0 += 1)
    for (int c1 = 1; c1 <= n; c1 += 1)
      s2(c0, c1);
} else if (n >= m + 1) {
  for (int c0 = 1; c0 <= n; c0 += 1)
    for (int c1 = 1; c1 <= n; c1 += 1)
      s0(c0, c1);
} else {
  for (int c0 = 1; c0 <= n; c0 += 1)
    for (int c1 = 1; c1 <= n; c1 += 1)
      s1(c0, c1);
}
