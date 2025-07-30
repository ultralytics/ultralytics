{
  for (int c0 = 0; c0 <= floord(n - 1, 3); c0 += 1)
    for (int c2 = 3 * c0 + 1; c2 <= min(n, 3 * c0 + 3); c2 += 1)
      S1(c2, c0);
  for (int c0 = floord(n, 3); c0 <= 2 * floord(n, 3); c0 += 1)
    for (int c1 = 0; c1 < n; c1 += 1)
      for (int c3 = max(1, (n % 3) - n + 3 * c0); c3 <= min(n, (n % 3) - n + 3 * c0 + 2); c3 += 1)
        S2(c1 + 1, c3, 0, n / 3, c0 - n / 3);
}
