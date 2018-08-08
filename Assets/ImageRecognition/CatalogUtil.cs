using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

public class CatalogItem
{
    public int Id { get; set; }
    public string Name { get; set; }
    public string DisplayName { get; set; }
}

public static class CatalogUtil
{
    // regexes with different new line symbols
    private static string CATALOG_ITEM_PATTERN = @"item {{{0}  name: ""(?<name>.*)""{0}  id: (?<id>\d+){0}  display_name: ""(?<displayName>.*)""{0}}}";
    private static string CATALOG_ITEM_PATTERN_ENV = string.Format(CultureInfo.InvariantCulture, CATALOG_ITEM_PATTERN, Environment.NewLine);
    private static string CATALOG_ITEM_PATTERN_UNIX = string.Format(CultureInfo.InvariantCulture, CATALOG_ITEM_PATTERN, "\n");

    public static IEnumerable<CatalogItem> ReadCatalogItems(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            yield break;
        }

        Regex regex = new Regex(CATALOG_ITEM_PATTERN_ENV);
        var matches = regex.Matches(text);
        if (matches.Count == 0)
        {
            regex = new Regex(CATALOG_ITEM_PATTERN_UNIX);
            matches = regex.Matches(text);
        }

        foreach (Match match in matches)
        {
            var name = match.Groups[1].Value;
            var id = int.Parse(match.Groups[2].Value);
            var displayName = match.Groups[3].Value;

            yield return new CatalogItem()
            {
                Id = id,
                Name = name,
                DisplayName = displayName
            };
        }
    }
}