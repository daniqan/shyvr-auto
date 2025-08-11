"""
Integration tests for The Graph Protocol client
Tests use REAL API calls - no mocks
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml_analysis.market_data import GraphProtocolClient, DeFiMetrics
from src.utils.system_secrets import get_system_secrets


class TestGraphProtocolClient:
    """Test suite for Graph Protocol client with real API calls"""
    
    @pytest.fixture
    async def client(self):
        """Create Graph Protocol client instance"""
        client = GraphProtocolClient()
        yield client
        await client.close()
    
    @pytest.fixture
    def api_keys(self):
        """Get API keys from system secrets"""
        secrets = get_system_secrets()
        return {
            'api_key': secrets.graph_api_key,
            'api_token': secrets.graph_api_token
        }
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, api_keys):
        """Test client initializes with proper API keys"""
        client = GraphProtocolClient()
        
        assert client is not None
        assert client.api_key is not None or client.api_token is not None
        assert "thegraph.com" in client.base_url or "studio" in client.base_url
        
        # Test known subgraph IDs are configured
        assert "uniswap_v3" in client.SUBGRAPH_IDS
        # Other subgraphs commented out until we find correct IDs
        # assert "aave_v3" in client.SUBGRAPH_IDS
        # assert "compound_v3" in client.SUBGRAPH_IDS
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_query_uniswap_tvl_current(self, client):
        """Test querying current TVL from Uniswap V3 subgraph"""
        # Query for current Uniswap V3 factory data
        query = """
        {
            uniswapDayDatas(first: 1, orderBy: date, orderDirection: desc) {
                date
                tvlUSD
                volumeUSD
                txCount
            }
        }
        """
        
        result = await client.query_subgraph("uniswap_v3", query)
        
        assert result is not None
        assert "uniswapDayDatas" in result
        assert len(result["uniswapDayDatas"]) > 0
        
        day_data = result["uniswapDayDatas"][0]
        assert "tvlUSD" in day_data
        assert "volumeUSD" in day_data
        assert "txCount" in day_data
        assert float(day_data["tvlUSD"]) > 0
    
    @pytest.mark.asyncio
    async def test_query_uniswap_historical_30_days(self, client):
        """Test querying 30 days of historical data from Uniswap"""
        # Calculate timestamp for 30 days ago
        thirty_days_ago = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
        
        query = """
        query($timestamp: Int!) {
            uniswapDayDatas(
                first: 30,
                orderBy: date,
                orderDirection: desc,
                where: { date_gte: $timestamp }
            ) {
                date
                tvlUSD
                volumeUSD
                txCount
                feesUSD
            }
        }
        """
        
        variables = {"timestamp": thirty_days_ago}
        result = await client.query_subgraph("uniswap_v3", query, variables)
        
        assert result is not None
        assert "uniswapDayDatas" in result
        assert len(result["uniswapDayDatas"]) >= 28  # Allow for some missing days
        
        # Verify data is properly ordered
        dates = [int(d["date"]) for d in result["uniswapDayDatas"]]
        assert dates == sorted(dates, reverse=True)
        
        # Verify all required fields are present
        for day_data in result["uniswapDayDatas"]:
            assert float(day_data["tvlUSD"]) > 0
            assert float(day_data["volumeUSD"]) >= 0
            assert int(day_data["txCount"]) >= 0
    
    @pytest.mark.skip(reason="Aave V3 subgraph ID needs to be verified")
    @pytest.mark.asyncio
    async def test_query_aave_v3_markets(self, client):
        """Test querying Aave V3 market data"""
        query = """
        {
            markets(first: 5, orderBy: totalValueLockedUSD, orderDirection: desc) {
                id
                name
                totalValueLockedUSD
                totalBorrowsUSD
                inputTokenBalance
                rates {
                    rate
                    type
                }
            }
        }
        """
        
        result = await client.query_subgraph("aave_v3", query)
        
        assert result is not None
        assert "markets" in result
        assert len(result["markets"]) > 0
        
        # Check first market has required fields
        market = result["markets"][0]
        assert "totalValueLockedUSD" in market
        assert float(market["totalValueLockedUSD"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_defi_metrics_historical(self, client):
        """Test high-level DeFi metrics retrieval for Uniswap"""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        metrics = await client.get_historical_defi_metrics(
            protocols=["uniswap_v3"],  # Only test Uniswap for now
            start_date=start_date,
            end_date=end_date,
            interval="daily"
        )
        
        assert metrics is not None
        assert len(metrics) >= 5  # At least 5 days of data (allowing for gaps)
        
        # Check each metric has required fields
        for metric in metrics:
            assert isinstance(metric, DeFiMetrics)
            assert metric.total_value_locked > 0
            assert metric.protocols_count >= 1  # At least Uniswap
            assert metric.timestamp is not None
    
    @pytest.mark.skip(reason="Compound V3 subgraph ID needs to be verified")
    @pytest.mark.asyncio
    async def test_get_compound_v3_supply_rates(self, client):
        """Test querying Compound V3 supply rates"""
        query = """
        {
            markets(first: 3) {
                id
                name
                supplyRate
                borrowRate
                totalSupply
                totalBorrow
                collateralFactor
            }
        }
        """
        
        result = await client.query_subgraph("compound_v3", query)
        
        assert result is not None
        # Note: Compound V3 subgraph structure may vary
        # Adjust assertions based on actual response
        if "markets" in result and len(result["markets"]) > 0:
            market = result["markets"][0]
            assert "id" in market
    
    @pytest.mark.asyncio
    async def test_pagination_large_dataset(self, client):
        """Test pagination for large historical queries"""
        # Query with pagination
        all_results = []
        last_id = "0"
        
        for _ in range(3):  # Get 3 pages
            query = """
            query($lastId: String!) {
                pools(
                    first: 10,
                    where: { id_gt: $lastId },
                    orderBy: id
                ) {
                    id
                    totalValueLockedUSD
                    volumeUSD
                }
            }
            """
            
            variables = {"lastId": last_id}
            result = await client.query_subgraph("uniswap_v3", query, variables)
            
            if not result or "pools" not in result or len(result["pools"]) == 0:
                break
            
            all_results.extend(result["pools"])
            last_id = result["pools"][-1]["id"]
        
        assert len(all_results) > 0
        assert len(all_results) <= 30  # Maximum 3 pages * 10 items
        
        # Verify no duplicates
        ids = [r["id"] for r in all_results]
        assert len(ids) == len(set(ids))
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_subgraph(self, client):
        """Test error handling for invalid subgraph ID"""
        with pytest.raises(Exception) as exc_info:
            await client.query_subgraph("invalid_subgraph", "{ test }")
        
        assert "invalid_subgraph" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_query(self, client):
        """Test error handling for malformed GraphQL query"""
        # Malformed query (missing closing brace)
        bad_query = """
        {
            pools(first: 1
                id
            }
        """
        
        with pytest.raises(Exception) as exc_info:
            await client.query_subgraph("uniswap_v3", bad_query)
        
        # Should get a GraphQL syntax error
        assert "syntax" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_rate_limiting_compliance(self, client):
        """Test that client respects rate limits"""
        # Make multiple rapid requests
        queries = []
        for i in range(5):
            query = f"""
            {{
                pools(first: 1, skip: {i}) {{
                    id
                    totalValueLockedUSD
                }}
            }}
            """
            queries.append(client.query_subgraph("uniswap_v3", query))
        
        # All should complete without rate limit errors
        results = await asyncio.gather(*queries)
        assert len(results) == 5
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_multi_protocol_aggregation(self, client):
        """Test aggregating data from Uniswap protocol"""
        # Query TVL from Uniswap (only protocol with verified subgraph)
        protocols_tvl = {}
        
        # Uniswap V3
        uni_query = """
        {
            uniswapDayDatas(first: 1, orderBy: date, orderDirection: desc) {
                tvlUSD
            }
        }
        """
        uni_result = await client.query_subgraph("uniswap_v3", uni_query)
        if uni_result and "uniswapDayDatas" in uni_result:
            protocols_tvl["uniswap"] = float(uni_result["uniswapDayDatas"][0]["tvlUSD"])
        
        # Skip other protocols until we have correct subgraph IDs
        # # Aave V3
        # aave_query = """
        # {
        #     markets(first: 100) {
        #         totalValueLockedUSD
        #     }
        # }
        # """
        # aave_result = await client.query_subgraph("aave_v3", aave_query)
        # if aave_result and "markets" in aave_result:
        #     protocols_tvl["aave"] = sum(float(m["totalValueLockedUSD"]) for m in aave_result["markets"])
        
        # Verify we got data from at least one protocol
        assert len(protocols_tvl) > 0
        assert sum(protocols_tvl.values()) > 0
        
        # Calculate total DeFi TVL
        total_tvl = sum(protocols_tvl.values())
        assert total_tvl > 1_000_000  # Should be at least $1M
    
    @pytest.mark.asyncio
    async def test_historical_data_consistency(self, client):
        """Test that historical data is consistent and complete"""
        # Get 7 days of historical data
        seven_days_ago = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp())
        
        query = """
        query($timestamp: Int!) {
            uniswapDayDatas(
                first: 7,
                orderBy: date,
                orderDirection: asc,
                where: { date_gte: $timestamp }
            ) {
                date
                tvlUSD
                volumeUSD
            }
        }
        """
        
        variables = {"timestamp": seven_days_ago}
        result = await client.query_subgraph("uniswap_v3", query, variables)
        
        assert result is not None
        assert "uniswapDayDatas" in result
        
        day_datas = result["uniswapDayDatas"]
        assert len(day_datas) >= 5  # At least 5 days of data
        
        # Check data consistency
        for i in range(1, len(day_datas)):
            current = day_datas[i]
            previous = day_datas[i-1]
            
            # Dates should be sequential (allowing for gaps)
            date_diff = int(current["date"]) - int(previous["date"])
            assert date_diff >= 86400  # At least 1 day apart
            assert date_diff <= 172800  # At most 2 days apart (allowing for gaps)
            
            # TVL changes should be reasonable (not 0 and not 100x changes)
            tvl_ratio = float(current["tvlUSD"]) / float(previous["tvlUSD"])
            assert 0.5 < tvl_ratio < 2.0  # TVL shouldn't change by more than 2x daily


@pytest.mark.asyncio
async def test_integration_with_corpus_collector():
    """Test that Graph data can be integrated into corpus collector"""
    from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector
    
    # This test verifies the integration point
    collector = InitialCorpusCollector()
    
    # Verify Graph client is available
    assert hasattr(collector, 'graph_client') or hasattr(collector, '_init_graph_client')
    
    # Test collection with Graph data
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=1)
    
    # Note: This is a placeholder - actual implementation will be done after client is ready
    # The test ensures the integration point exists